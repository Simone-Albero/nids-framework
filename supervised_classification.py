import logging
import pickle
import configparser

import pandas as pd
from rich.logging import RichHandler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from nids_framework.data import (
    properties,
    processor,
    utilities,
    transformation_builder,
    samplers,
    tabular_datasets,
)
from nids_framework.model import transformer
from nids_framework.training import trainer, metrics


def supervised_classification(epoch, epoch_steps, metric_path = "logs/binary_metrics.csv", isBinary = True, seed = 42):
    PROPERTIES_PATH = "configs/dataset_properties.ini"

    # DATASET_NAME = "nf_ton_iot_v2_anonymous"
    DATASET_NAME = "nf_unsw_nb15_v2_anonymous"
    # DATASET_NAME = "cse_cic_ids_2018_v2_anonymous"
    CONFIG_PATH = "configs/config.ini"
    CONFIG_NAME = "small"

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    config = config[CONFIG_NAME]

    CATEGORICAL_LEVELS = int(config['categorical_levels'])
    BOUND = int(config['bound'])

    BATCH_SIZE = int(config['batch_size'])
    WINDOW_SIZE = int(config['window_size'])
    LATENT_DIM = int(config['latent_dim'])
    NUM_HEADS = int(config['num_heads'])
    NUM_LAYERS = int(config['num_layers'])
    DROPOUT = float(config['dropout'])
    FF_DIM = int(config['ff_dim'])
    LR = float(config['lr'])
    WEIGHT_DECAY = float(config['weight_decay'])

    N_EPOCH = epoch
    EPOCH_STEPS = epoch_steps

    named_prop = properties.NamedDatasetProperties(PROPERTIES_PATH)
    prop = named_prop.get_properties(DATASET_NAME)

    df_train = pd.read_csv(prop.train_path)
    df_test = pd.read_csv(prop.test_path)

    trans_builder = transformation_builder.TransformationBuilder()

    min_values, max_values = utilities.min_max_values(df_train, prop, BOUND)
    unique_values = utilities.unique_values(df_train, prop, CATEGORICAL_LEVELS)

    class_mapping = None
    if not isBinary:
        class_mapping, _ = utilities.labels_mapping(df_train, prop)
        logging.info(f"Class Mapping:\n {class_mapping}\n")


    @trans_builder.add_step(order=1)
    def base_pre_processing(dataset):
        return utilities.base_pre_processing(dataset, prop, BOUND)

    @trans_builder.add_step(order=2)
    def log_pre_processing(dataset):
        return utilities.log_pre_processing(dataset, prop, min_values, max_values)

    @trans_builder.add_step(order=3)
    def categorical_conversion(dataset):
        return utilities.categorical_pre_processing(
            dataset, prop, unique_values, CATEGORICAL_LEVELS
        )
    
    if isBinary:
        @trans_builder.add_step(order=4)
        def binary_label_conversion(dataset):
            return utilities.binary_benign_label_conversion(dataset, prop)
    else:
        @trans_builder.add_step(order=4)
        def binary_label_conversion(dataset):
            return utilities.multi_class_label_conversion(dataset, prop, class_mapping)

    @trans_builder.add_step(order=5)
    def split_data_for_torch(dataset):
        return utilities.split_data_for_torch(dataset, prop)

    transformations = trans_builder.build()

    proc = processor.Processor(transformations)
    X_train, y_train = proc.apply(df_train)
    X_test, y_test = proc.apply(df_test)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = tabular_datasets.TabularDataset(
        X_train[prop.numeric_features],
        X_train[prop.categorical_features],
        y_train,
        device,
    )

    test_dataset = tabular_datasets.TabularDataset(
        X_test[prop.numeric_features], X_test[prop.categorical_features], y_test, device
    )

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEVELS):
        return utilities.one_hot_encoding(sample, categorical_levels)

    transformations = trans_builder.build()
    train_dataset.set_categorical_transformation(transformations)
    test_dataset.set_categorical_transformation(transformations)

    train_sampler = samplers.RandomSlidingWindowSampler(
        train_dataset, window_size=WINDOW_SIZE, seed = seed
    )
    test_sampler = samplers.RandomSlidingWindowSampler(
        test_dataset, window_size=WINDOW_SIZE, seed = seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        drop_last=True,
        shuffle=False,
    )

    input_dim = next(iter(train_dataloader))[0].shape[-1]
    logging.info(f"Input dim: {input_dim}")

    model = transformer.TransformerClassifier(
        output_dim=1 if class_mapping is None else len(class_mapping),
        input_dim=input_dim,
        model_dim=LATENT_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        seq_length=WINDOW_SIZE
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of parameters: {total_params}")

    class_proportions = y_train.value_counts(normalize=True).sort_index()
    if isBinary:
        pos_weight = torch.tensor(class_proportions.iloc[0] / class_proportions.iloc[1], dtype=torch.float32, device=device)
        logging.info(f"pos_weight: {pos_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        class_weights = torch.tensor(1.0 / class_proportions.values, dtype=torch.float32, device=device)
        logging.info(f"class_weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)


    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train = trainer.Trainer(model, criterion, optimizer)

    train.train(
        n_epoch=N_EPOCH,
        train_data_loader=train_dataloader,
        epoch_steps=EPOCH_STEPS,
    )
    model.save_model_weights(f"saves/baseline/{DATASET_NAME}/{EPOCH_STEPS}.pt")

    if isBinary:
        metric = metrics.BinaryClassificationMetric()
    else:
        metric = metrics.MulticlassClassificationMetric(len(class_mapping))

    train.test(test_dataloader, metric)
    metric.save(metric_path)

if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    # supervised_classification(1, 200, "logs/baseline.csv", True, 42)

    for seed in [42, 29, 13, 3, 15]:
        for i in range(1, 15, 1):
            supervised_classification(1, 25*i, f"logs/{seed}_baseline.csv", True, seed)

