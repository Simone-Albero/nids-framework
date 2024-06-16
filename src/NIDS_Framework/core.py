import logging
from rich.logging import RichHandler

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import (
    processor,
    utilities,
    transformation_builder,
    samplers,
    tabular_datasets,
)
from utilities import trace_stats
from model import nn_classifier
from training import (
    trainer,
    metrics,
)


def data_loader_test():
    prop = processor.DatasetProperties(
        features=[
            "NUM_PKTS_UP_TO_128_BYTES",
            "SRC_TO_DST_SECOND_BYTES",
            "OUT_PKTS",
            "OUT_BYTES",
            "NUM_PKTS_128_TO_256_BYTES",
            "DST_TO_SRC_AVG_THROUGHPUT",
            "DURATION_IN",
            "L4_SRC_PORT",
            "ICMP_TYPE",
            "PROTOCOL",
            "SERVER_TCP_FLAGS",
            "IN_PKTS",
            "NUM_PKTS_512_TO_1024_BYTES",
            "CLIENT_TCP_FLAGS",
            "TCP_WIN_MAX_IN",
            "NUM_PKTS_256_TO_512_BYTES",
            "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN",
            "LONGEST_FLOW_PKT",
            "L4_DST_PORT",
            "MIN_TTL",
            "DST_TO_SRC_SECOND_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES",
            "DURATION_OUT",
            "FLOW_DURATION_MILLISECONDS",
            "TCP_FLAGS",
            "MAX_TTL",
            "SRC_TO_DST_AVG_THROUGHPUT",
            "ICMP_IPV4_TYPE",
            "MAX_IP_PKT_LEN",
            "RETRANSMITTED_OUT_BYTES",
            "IN_BYTES",
            "RETRANSMITTED_IN_BYTES",
            "TCP_WIN_MAX_OUT",
            "L7_PROTO",
            "RETRANSMITTED_OUT_PKTS",
            "RETRANSMITTED_IN_PKTS",
        ],
        categorical_features=[
            "CLIENT_TCP_FLAGS",
            "L4_SRC_PORT",
            "TCP_FLAGS",
            "ICMP_IPV4_TYPE",
            "ICMP_TYPE",
            "PROTOCOL",
            "SERVER_TCP_FLAGS",
            "L4_DST_PORT",
            "L7_PROTO",
        ],
        labels="Attack",
        benign_label="Benign",
    )

    dataset_path = "dataset/NF-UNSW-NB15-v2.csv"
    df = pd.read_csv(dataset_path)
    proc = processor.Processor(df, prop)

    trans_builder = transformation_builder.TransformationBuilder()
    CATEGORICAL_LEV = 32
    BOUND = 1000000

    @trans_builder.add_step(order=1)
    # @trace_stats()
    def categorical_conversion(dataset, properties, categorical_levels=CATEGORICAL_LEV):
        utilities.categorical_pre_processing(dataset, properties, categorical_levels)

    @trans_builder.add_step(order=2)
    def bynary_label_conversion(dataset, properties):
        utilities.bynary_label_conversion(dataset, properties)

    proc.transformations = trans_builder.build()
    X, y = proc.fit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_vaild, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    train_dataset = tabular_datasets.TabularDataset(
        X_train[prop.numeric_features], X_train[prop.categorical_features], y_train
    )
    valid_dataset = tabular_datasets.TabularDataset(
        X_vaild[prop.numeric_features], X_vaild[prop.categorical_features], y_valid
    )
    test_dataset = tabular_datasets.TabularDataset(
        X_test[prop.numeric_features], X_test[prop.categorical_features], y_test
    )

    @trans_builder.add_step(order=1)
    def bound_trans(sample, bound=BOUND):
        return utilities.bound_transformation(sample, bound)

    @trans_builder.add_step(order=2)
    def log_trans(sample):
        return utilities.log_transformation(sample)

    train_dataset.numeric_transformation = trans_builder.build()

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)

    train_dataset.categorical_transformation = trans_builder.build()

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)

    valid_dataset.categorical_transformation = trans_builder.build()

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)

    test_dataset.categorical_transformation = trans_builder.build()

    WINDOW_SIZE = 8
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPUT = 0.4
    LR = 0.001
    MOMENTUM = 0.9
    WHIGHT_DECAY = 0.1

    train_sampler = samplers.FairSlidingWindowSampler(
        train_dataset, y_train, window_size=WINDOW_SIZE
    )
    valid_sampler = samplers.RandomSlidingWindowSampler(
        valid_dataset, window_size=WINDOW_SIZE
    )
    test_sampler = samplers.RandomSlidingWindowSampler(
        test_dataset, window_size=WINDOW_SIZE
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        drop_last=True,
        shuffle=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=64,
        sampler=valid_sampler,
        drop_last=True,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, sampler=test_sampler, drop_last=True, shuffle=False
    )

    inputs, _ = next(iter(train_dataloader))
    input_shape = inputs.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn_classifier.NNClassifier(
        input_shape[-1], num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPUT
    ).to(device=device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WHIGHT_DECAY
    )

    N_EPOCH = 100
    EPOCH_STEPS = 128
    EPOCH_UNTI_VALIDATION = 5
    PATIENCE = 3
    DELTA = 0.05

    train = trainer.Trainer(model, criterion, optimizer)
    train.train(
        n_epoch=N_EPOCH,
        train_data_loader=train_dataloader,
        epoch_steps=EPOCH_STEPS,
        epochs_until_validation=EPOCH_UNTI_VALIDATION,
        valid_data_loader=valid_dataloader,
        patience=PATIENCE,
        delta=DELTA,
    )

    pred_func = lambda x: torch.where(x >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
    metric = metrics.BinaryClassificationMetric()
    train.test(test_dataloader, pred_func, metric)
    train.save_model()


def main():
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )
    data_loader_test()


if __name__ == "__main__":
    main()
