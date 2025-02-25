import logging
import os

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from rich.logging import RichHandler

from nids_framework.data.config_manager import ConfigManager
from nids_framework.data.properties import DatasetProperties
from nids_framework.data.transforms import (
    clipper,
    frequency_encoder,
    log_scaler,
    label_encoder,
)
from nids_framework.data.utilities import (
    label_mapping,
    binary_label_mapping,
    OneHotEncoder,
)
from nids_framework.data.tabular_datasets import TabularDataset
from nids_framework.data.samplers import RandomSlidingWindowSampler
from nids_framework.model.autoencoder import TransformerAutoencoder
from nids_framework.model.loss import HybridReconstructionLoss
from nids_framework.training.trainer import Trainer
from nids_framework.model.classifier import TransformerClassifier
from nids_framework.training.metrics.binary_classification import (
    BinaryClassificationMetric,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )


def load_dataset_properties(dataset_name):
    return DatasetProperties.from_config(f"configs/{dataset_name}.json")


def static_preprocessing(dataset_properties, is_binary=True, config_tag="small"):
    logging.info("Loading dataset...")
    df_train = pd.read_csv(dataset_properties.train_path)
    df_test = pd.read_csv(dataset_properties.test_path)

    mapping = (
        binary_label_mapping(
            df_train[dataset_properties.label], dataset_properties.benign_label
        )
        if is_binary
        else label_mapping(df_train[dataset_properties.label])
    )

    numeric_pipeline = Pipeline(
        [
            (
                "clip",
                clipper.Clipper(
                    dataset_properties, ConfigManager.get_value(config_tag, "border")
                ),
            ),
            ("log_scale", log_scaler.LogScaler(dataset_properties)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", numeric_pipeline, dataset_properties.numeric_features),
            (
                "frequency_encoding",
                frequency_encoder.FrequencyEncoder(
                    dataset_properties,
                    ConfigManager.get_value(config_tag, "categorical_levels"),
                ),
                dataset_properties.categorical_features,
            ),
            (
                "label_encoding",
                label_encoder.LabelEncoder(dataset_properties, mapping),
                [dataset_properties.label],
            ),
        ],
        verbose=True,
    )

    logging.info("Applying preprocessing...")
    df_train = pd.DataFrame(
        preprocessor.fit_transform(df_train),
        columns=dataset_properties.features + [dataset_properties.label],
    )
    df_test = pd.DataFrame(
        preprocessor.transform(df_test),
        columns=dataset_properties.features + [dataset_properties.label],
    )

    return (
        df_train[dataset_properties.features],
        df_train[dataset_properties.label],
        df_test[dataset_properties.features],
        df_test[dataset_properties.label],
    )


def get_data_loader(
    dataset_properties, x, y=None, device="cpu", config_tag="small", seed=42
):
    num_workers = min(
        int(os.cpu_count() / 2), ConfigManager.get_value(config_tag, "batch_size")
    )

    dataset = TabularDataset(
        x[dataset_properties.numeric_features],
        x[dataset_properties.categorical_features],
        y,
        device="cpu",
    )

    one_hot_encoder = OneHotEncoder(
        ConfigManager.get_value(config_tag, "categorical_levels")
    )

    dataset.set_transforms({"categorical": [one_hot_encoder]})

    sampler = RandomSlidingWindowSampler(
        dataset,
        window_size=ConfigManager.get_value(config_tag, "window_size"),
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=ConfigManager.get_value(config_tag, "batch_size"),
        sampler=sampler,
        drop_last=True,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def train_model(
    model,
    train_dataloader,
    criterion,
    config_tag="small",
    epochs=1,
    epoch_steps=200,
    device="cpu",
):
    optimizer = optim.Adam(
        model.parameters(),
        lr=ConfigManager.get_value(config_tag, "lr"),
        weight_decay=ConfigManager.get_value(config_tag, "weight_decay"),
    )
    trainer = Trainer(criterion, optimizer, device=device)
    trainer.set_model(model)
    trainer.train(n_epoch=epochs, data_loader=train_dataloader, epoch_steps=epoch_steps)


def test_model(
    model,
    test_dataloader,
    criterion,
    metric,
    metric_path="logs/binary_metrics.csv",
    device="cpu",
):
    trainer = Trainer(criterion, device=device)
    trainer.set_model(model)
    trainer.test(test_dataloader, metric)
    metric.save(metric_path)


def run_experiment(
    dataset_properties,
    x_train,
    y_train,
    x_test,
    y_test,
    finetuning_steps=100,
    pretraining_steps=1800,
    config_tag="small",
    metric_path="logs/hybrid",
    seed=42,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_data_loader = get_data_loader(
        dataset_properties, x_train, device=device, seed=seed
    )
    input_dim = next(iter(training_data_loader))[0].shape[-1]

    autoencoder = TransformerAutoencoder(
        input_dim=input_dim,
        model_dim=ConfigManager.get_value(config_tag, "latent_dim"),
        num_heads=ConfigManager.get_value(config_tag, "num_heads"),
        num_layers=ConfigManager.get_value(config_tag, "num_layers"),
        ff_dim=ConfigManager.get_value(config_tag, "ff_dim"),
        dropout=ConfigManager.get_value(config_tag, "dropout"),
        numeric_dim=len(dataset_properties.numeric_features),
        categorical_dim=input_dim - len(dataset_properties.numeric_features),
        noise_factor=ConfigManager.get_value(config_tag, "noise_factor"),
    ).to(device)

    criterion = HybridReconstructionLoss()
    train_model(
        autoencoder,
        training_data_loader,
        criterion,
        epoch_steps=pretraining_steps,
        device=device,
    )

    training_data_loader = get_data_loader(
        dataset_properties, x_train, y_train, device=device, seed=seed
    )
    test_data_loader = get_data_loader(
        dataset_properties, x_test, y_test, device=device, seed=seed
    )

    class_proportions = y_train.value_counts(normalize=True).sort_index()
    pos_weight = torch.tensor(
        class_proportions.iloc[0] / class_proportions.iloc[1],
        dtype=torch.float32,
        device=device,
    )
    logging.info(f"pos_weight: {pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # class_weights = torch.tensor(1.0 / class_proportions.values, dtype=torch.float32, device=device)
    # logging.info(f"class_weights: {class_weights}")
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    pretrained_classifier = TransformerClassifier(
        output_dim=1,
        input_dim=input_dim,
        model_dim=ConfigManager.get_value(config_tag, "latent_dim"),
        num_heads=ConfigManager.get_value(config_tag, "num_heads"),
        num_layers=ConfigManager.get_value(config_tag, "num_layers"),
        ff_dim=ConfigManager.get_value(config_tag, "ff_dim"),
        dropout=ConfigManager.get_value(config_tag, "dropout"),
    ).to(device)

    pretrained_classifier.encoder = autoencoder.encoder
    pretrained_classifier.embedding = autoencoder.embedding

    train_model(
        pretrained_classifier,
        training_data_loader,
        criterion,
        config_tag=config_tag,
        epochs=1,
        epoch_steps=finetuning_steps,
        device=device,
    )
    metric = BinaryClassificationMetric()
    test_model(
        pretrained_classifier,
        test_data_loader,
        criterion,
        metric,
        metric_path=f"{metric_path}/{seed}.csv",
        device=device,
    )


def run_baseline(
    dataset_properties,
    x_train,
    y_train,
    x_test,
    y_test,
    finetuning_steps=100,
    config_tag="small",
    metric_path="logs/baseline",
    seed=42,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_data_loader = get_data_loader(
        dataset_properties, x_train, y_train, device=device, seed=seed
    )
    test_data_loader = get_data_loader(
        dataset_properties, x_test, y_test, device=device, seed=seed
    )
    input_dim = next(iter(training_data_loader))[0].shape[-1]

    class_proportions = y_train.value_counts(normalize=True).sort_index()
    pos_weight = torch.tensor(
        class_proportions.iloc[0] / class_proportions.iloc[1],
        dtype=torch.float32,
        device=device,
    )
    logging.info(f"pos_weight: {pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    baseline_classifier = TransformerClassifier(
        output_dim=1,
        input_dim=input_dim,
        model_dim=ConfigManager.get_value(config_tag, "latent_dim"),
        num_heads=ConfigManager.get_value(config_tag, "num_heads"),
        num_layers=ConfigManager.get_value(config_tag, "num_layers"),
        ff_dim=ConfigManager.get_value(config_tag, "ff_dim"),
        dropout=ConfigManager.get_value(config_tag, "dropout"),
    ).to(device)

    train_model(
        baseline_classifier,
        training_data_loader,
        criterion,
        config_tag=config_tag,
        epochs=1,
        epoch_steps=finetuning_steps,
        device=device,
    )
    metric = BinaryClassificationMetric()
    test_model(
        baseline_classifier,
        test_data_loader,
        criterion,
        metric,
        metric_path=f"{metric_path}/{seed}.csv",
        device=device,
    )


if __name__ == "__main__":
    dataset_name = "nf_unsw_nb15_v2"
    # dataset_name = "nf_ton_iot_v2"
    # dataset_name = "cse_cic_ids_2018_v2"

    setup_logging()
    dataset_properties = load_dataset_properties(dataset_name)
    ConfigManager.load_config("configs/parameter.json")

    config_tag = "small"
    x_train, y_train, x_test, y_test = static_preprocessing(
        dataset_properties, config_tag=config_tag
    )

    seeds = [42, 29, 13, 3, 8]

    for seed in seeds:
        for i in range(1, 26, 1):
            run_experiment(
                dataset_properties,
                x_train,
                y_train,
                x_test,
                y_test,
                finetuning_steps=25,
                pretraining_steps=100 * i,
                config_tag=config_tag,
                metric_path=f"logs/{dataset_name}/pretrain_25b",
                seed=seed,
            )

        for i in range(1, 31, 1):
            run_experiment(
                dataset_properties,
                x_train,
                y_train,
                x_test,
                y_test,
                finetuning_steps=10 * i,
                pretraining_steps=1800,
                config_tag=config_tag,
                metric_path=f"logs/{dataset_name}/hybrid_10b",
                seed=seed,
            )
            run_baseline(
                dataset_properties,
                x_train,
                y_train,
                x_test,
                y_test,
                finetuning_steps=10 * i,
                config_tag=config_tag,
                metric_path=f"logs/{dataset_name}/baseline_10b",
                seed=seed,
            )
