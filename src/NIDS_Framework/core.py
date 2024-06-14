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
from training import trainer


def data_loader_test():
    prop = processor.DatasetProperties(
        features=[
            'NUM_PKTS_UP_TO_128_BYTES',
            'SRC_TO_DST_SECOND_BYTES',
            'OUT_PKTS',
            'OUT_BYTES',
            'NUM_PKTS_128_TO_256_BYTES',
            'DST_TO_SRC_AVG_THROUGHPUT',
            'DURATION_IN',
            'L4_SRC_PORT',
            'ICMP_TYPE',
            'PROTOCOL',
            'SERVER_TCP_FLAGS',
            'IN_PKTS',
            'NUM_PKTS_512_TO_1024_BYTES',
            'CLIENT_TCP_FLAGS',
            'TCP_WIN_MAX_IN',
            'NUM_PKTS_256_TO_512_BYTES',
            'SHORTEST_FLOW_PKT',
            'MIN_IP_PKT_LEN',
            'LONGEST_FLOW_PKT',
            'L4_DST_PORT',
            'MIN_TTL',
            'DST_TO_SRC_SECOND_BYTES',
            'NUM_PKTS_1024_TO_1514_BYTES',
            'DURATION_OUT',
            'FLOW_DURATION_MILLISECONDS',
            'TCP_FLAGS',
            'MAX_TTL',
            'SRC_TO_DST_AVG_THROUGHPUT',
            'ICMP_IPV4_TYPE',
            'MAX_IP_PKT_LEN',
            'RETRANSMITTED_OUT_BYTES',
            'IN_BYTES',
            'RETRANSMITTED_IN_BYTES',
            'TCP_WIN_MAX_OUT',
            'L7_PROTO',
            'RETRANSMITTED_OUT_PKTS',
            'RETRANSMITTED_IN_PKTS',
        ],
        categorical_features=[
            'CLIENT_TCP_FLAGS',
            'L4_SRC_PORT',
            'TCP_FLAGS',
            'ICMP_IPV4_TYPE',
            'ICMP_TYPE',
            'PROTOCOL',
            'SERVER_TCP_FLAGS',
            'L4_DST_PORT',
            'L7_PROTO',
        ],
        labels='Attack',
        benign_label='Benign',
    )

    dataset_path = 'dataset/NF-UNSW-NB15-v2.csv'
    df = pd.read_csv(dataset_path)
    proc = processor.Processor(df, prop)

    trans_builder = transformation_builder.TransformationBuilder()

    @trans_builder.add_step(order=1)
    #@trace_stats()
    def categorical_conversion(dataset, properties, categorical_levels=32):
        utilities.categorical_pre_processing(dataset, properties, categorical_levels)

    @trans_builder.add_step(order=2)
    def bynary_label_conversion(dataset, properties):
        utilities.bynary_label_conversion(dataset, properties)

    proc.transformations = trans_builder.build()
    X, y = proc.fit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = tabular_datasets.TabularDataset(
        X_train[prop.numeric_features], X_train[prop.categorical_features], y_train
    )
    test_dataset = tabular_datasets.TabularDataset(
        X_test[prop.numeric_features], X_test[prop.categorical_features], y_test
    )

    @trans_builder.add_step(order=1)
    def bound_trans(sample, bound=1000000):
        return utilities.bound_transformation(sample, bound)

    @trans_builder.add_step(order=2)
    def log_trans(sample):
        return utilities.log_transformation(sample)

    train_dataset.numeric_transformation = trans_builder.build()

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=32):
        return utilities.one_hot_encoding(sample, categorical_levels)

    train_dataset.categorical_transformation = trans_builder.build()

    train_sampler = samplers.FairSlidingWindowSampler(
        train_dataset, y_train, window_size=8
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, sampler=train_sampler, drop_last=True, shuffle=False
    )

    inputs, _ = next(iter(train_dataloader))
    input_shape = inputs.shape

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn_classifier.NNClassifier(input_shape[-1]).to(device=device)

    criterion = nn.BCELoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train = trainer.Trainer(model, criterion, optimizer)
    train.fit(3, train_dataloader, 128)

def main():
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format='%(message)s',
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )
    data_loader_test()

if __name__ == '__main__':
    main()
