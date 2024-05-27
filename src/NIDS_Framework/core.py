import logging
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from data_preparation import transformations, data_manager


def data_loader_test():
    prop = data_manager.DatasetProperties(
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
    logging.info(
        f"Tot features: {len(prop.features)}, Numeric features: {len(prop.numeric_features)}, Categorical features: {len(prop.categorical_features)}"
    )

    dataset_path = "dataset/NF-UNSW-NB15-v2.csv"
    dm = data_manager.DataManager(dataset_path, prop)

    bound = 100000

    @dm.numeric_transformation(priority=1)
    def task1(sample, bound=bound):
        return transformations.bound_transformation(sample, bound)

    @dm.numeric_transformation(priority=2)
    def task2(sample, bound=bound):
        return transformations.log_transformation(sample, bound)

    @dm.categorical_transformation(priority=1)
    def task3(sample, categorical_bound=32):
        return transformations.one_hot_transformation(sample, categorical_bound)

    train_dataset = dm.train_data()

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    for batch in train_dataloader:
        features, label = batch
        print(features[0].shape)
        #print(label.shape)
        break


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
