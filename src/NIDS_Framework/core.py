import logging

from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_preparation import (
    random_sw_sampler,
    tabular_dataset,
    processor,
    utilities,
    transformation_builder,
)


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
    proc = processor.Processor(dataset_path, prop, True)

    @proc.add_step(order=1)
    def categorical_conversion(dataset, properties, categorical_levels=32):
        utilities.categorical_pre_processing(dataset, properties, categorical_levels)

    X, y = proc.fit()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = tabular_dataset.TabularDataset(
        X_train[prop.numeric_features], X_train[prop.categorical_features], y_train
    )
    test_dataset = tabular_dataset.TabularDataset(
        X_test[prop.numeric_features], X_test[prop.categorical_features], y_test
    )

    trans_builder = transformation_builder.TransformationBuilder()

    @trans_builder.add_step(order=1)
    def bound_trans(sample, bound=1000000):
        return utilities.bound_transformation(sample, bound)

    @trans_builder.add_step(order=2)
    def log_trans(sample, bound=1000000):
        return utilities.log_transformation(sample, bound)

    numeric_transform = trans_builder.build()

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=32):
        return utilities.one_hot_encoding(sample, categorical_levels)

    categorical_transform = trans_builder.build()

    train_dataset.numeric_transformation = numeric_transform
    train_dataset.categorical_transformation = categorical_transform

    train_sampler = random_sw_sampler.RandomSlidingWindowSampler(
        train_dataset, window_size=8
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, sampler=train_sampler, drop_last=True
    )

    for batch in train_dataloader:
        features, labels = batch
        print(features.shape, labels.shape)
        break

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
