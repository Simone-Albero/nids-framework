from pipeline import Pipeline
from dataset import Dataset
from dataset import DatasetProperties


def pipeline_test():
    pipeline = Pipeline()

    @pipeline.register(priority=2)
    def task1():
        print("Task 1")

    @pipeline.register(priority=1)
    def task2():
        print("Task 2")

    @pipeline.register(priority=3)
    def task3():
        print("Task 3")
    
    pipeline.execute()

def dataset_test():
    properties = DatasetProperties(
        features=['NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES', 'OUT_PKTS', 'OUT_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN', 'L4_SRC_PORT', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'IN_PKTS', 'NUM_PKTS_512_TO_1024_BYTES', 'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN', 'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'LONGEST_FLOW_PKT', 'L4_DST_PORT', 'MIN_TTL', 'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS', 'TCP_FLAGS', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE', 'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES', 'IN_BYTES', 'RETRANSMITTED_IN_BYTES', 'TCP_WIN_MAX_OUT', 'L7_PROTO', 'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS'],
        categorical_features=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', 'TCP_FLAGS', 'ICMP_IPV4_TYPE', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'L4_DST_PORT', 'L7_PROTO'],
        labels_column="Attack",
        benign_label="Benign"
    )
    dataset_path = 'dataset/NF-UNSW-NB15-v2.csv'
    cache_path = 'dataset/cache/NF-UNSW-NB15.pk1'

    dataset = Dataset(cache_path, dataset_path, properties)
    print(dataset._df.head())

if __name__ == "__main__":
    dataset_test()

