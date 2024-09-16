import logging

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
from nids_framework.model import transformer, loss
from nids_framework.training import trainer, metrics


def self_supervised_classification():
    CONFIG_PATH = "configs/dataset_properties.ini"
    DATASET_NAME = "nf_ton_iot_v2_binary_anonymous"
    TRAIN_PATH = "datasets/NF-ToN-IoT-V2/NF-ToN-IoT-V2-Train.csv"
    TEST_PATH = "datasets/NF-ToN-IoT-V2/NF-ToN-IoT-V2-Test.csv"

    CATEGORICAL_LEV = 32
    BOUND = 100000000

    BATCH_SIZE = 64
    WINDOW_SIZE = 8
    EMBED_DIM = 256
    NUM_HEADS = 2
    NUM_LAYERS = 4
    DROPOUT = 0.1
    FF_DIM = 128
    LR = 0.0005
    WHIGHT_DECAY = 0.001

    N_EPOCH = 1
    EPOCH_STEPS = 2000
    # EPOCH_UNTIL_VALIDATION = 100
    # PATIENCE = 2
    # DELTA = 0.01

    named_prop = properties.NamedDatasetProperties(CONFIG_PATH)
    prop = named_prop.get_properties(DATASET_NAME)

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH, nrows=100000)

    trans_builder = transformation_builder.TransformationBuilder()

    min_values, max_values = utilities.min_max_values(df_train, prop, BOUND)
    unique_values = utilities.unique_values(df_train, prop, CATEGORICAL_LEV)

    @trans_builder.add_step(order=1)
    def base_pre_processing(dataset, properties):
        utilities.base_pre_processing(dataset, properties, BOUND)

    @trans_builder.add_step(order=2)
    def log_pre_processing(dataset, properties):
        utilities.log_pre_processing(dataset, properties, min_values, max_values)

    @trans_builder.add_step(order=3)
    def categorical_conversion(dataset, properties):
        utilities.categorical_pre_processing(
            dataset, properties, unique_values, CATEGORICAL_LEV
        )

    @trans_builder.add_step(order=4)
    def binary_label_conversion(dataset, properties):
        utilities.binary_label_conversion(dataset, properties)

    transformations = trans_builder.build()

    proc = processor.Processor(df_train, prop)
    proc.transformations = transformations
    proc.apply()
    X_train, y_train = proc.build()

    proc = processor.Processor(df_test, prop)
    proc.transformations = transformations
    proc.apply()
    X_test, y_test = proc.build()


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    train_dataset = tabular_datasets.TabularReconstructionDataset(
        X_train[prop.numeric_features],
        X_train[prop.categorical_features],
        device
    )

    test_dataset = tabular_datasets.TabularReconstructionDataset(
        X_test[prop.numeric_features], 
        X_test[prop.categorical_features], 
        device
    )

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)

    transformations = trans_builder.build()
    train_dataset.set_categorical_transformation(transformations)
    test_dataset.set_categorical_transformation(transformations)

    @trans_builder.add_step(order=1)
    def mask_features(sample):
        return utilities.mask_features(sample, 0.4)

    transformations = trans_builder.build()
    train_dataset.set_masking_transformation(transformations)
    test_dataset.set_masking_transformation(transformations)

    train_sampler = samplers.RandomSlidingWindowSampler(
        train_dataset, window_size=WINDOW_SIZE
    )
    test_sampler = samplers.RandomSlidingWindowSampler(
        test_dataset, window_size=WINDOW_SIZE
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

    input_shape = next(iter(train_dataloader))[0].shape[-1]

    model = transformer.TransformerAutoencoder(
        input_dim=input_shape,
        border=train_dataset.get_border(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of parameters: {total_params}")

    criterion = loss.HybridReconstructionLoss(test_dataset.get_border())
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WHIGHT_DECAY,
    )

    train = trainer.Trainer(model, criterion, optimizer)
    train.train(
        n_epoch=N_EPOCH,
        train_data_loader=train_dataloader,
        epoch_steps=EPOCH_STEPS,
    )
    train.save_model(f"saves/s{WINDOW_SIZE}.pt")
    #train.test(test_dataloader)


    train_dataset = tabular_datasets.TabularDataset(
        X_train[prop.numeric_features],
        X_train[prop.categorical_features],
        y_train,
        device
    )

    test_dataset = tabular_datasets.TabularDataset(
        X_test[prop.numeric_features], 
        X_test[prop.categorical_features], 
        y_test,
        device
    )

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)
    
    transformations = trans_builder.build()
    train_dataset.set_categorical_transformation(transformations)
    test_dataset.set_categorical_transformation(transformations)

    train_sampler = samplers.RandomSlidingWindowSampler(
        train_dataset, window_size=WINDOW_SIZE
    )
    test_sampler = samplers.RandomSlidingWindowSampler(
        test_dataset, window_size=WINDOW_SIZE
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

    input_shape = next(iter(train_dataloader))[0].shape[-1]
    
    pre_trained_encoder = model.encoder
    model = transformer.TransformerClassifier(
        num_classes=1,
        input_dim=input_shape,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ).to(device)
    
    for param in pre_trained_encoder.parameters():
        param.requires_grad = False
    model.encoder = pre_trained_encoder

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of parameters: {total_params}")

    class_proportions = y_train.value_counts(normalize=True).sort_index()
    weights = 1.0 / class_proportions.values
    weights = weights / weights.sum()
    logging.info(f"weights: {weights}")

    criterion = nn.BCELoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)[1])
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WHIGHT_DECAY,
    )

    train = trainer.Trainer(model, criterion, optimizer)
    train.set_model(model)
    
    train.train(
        n_epoch=1,
        train_data_loader=train_dataloader,
        epoch_steps=128,
    )
    train.save_model(f"saves/s{WINDOW_SIZE}.pt")

    metric = metrics.BinaryClassificationMetric()
    train.test(test_dataloader, metric)


def generate_train_test():
    df_path = "dataset/NF-ToN-IoT-V2/NF-ToN-IoT-V2.csv"
    df = pd.read_csv(df_path)

    train_size = int(len(df) * 0.85)
    test_size = len(df) - train_size

    df_train = df.head(train_size)
    df_test = df.tail(test_size)

    df_train.to_csv("dataset/NF-ToN-IoT-V2/NF-ToN-IoT-V2-Train.csv", index=False)
    df_test.to_csv("dataset/NF-ToN-IoT-V2/NF-ToN-IoT-V2-Test.csv", index=False)

if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    self_supervised_classification()