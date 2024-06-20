import logging
from rich.logging import RichHandler

import pandas as pd
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
from model import (
    nn_classifier,
    input_encoder,
    transformer,
    classification_head,
)
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
    BOUND = 1_000_000_000

    @trans_builder.add_step(order=1)
    def base_pre_processing(dataset, properties, train_mask):
        utilities.base_pre_processing(dataset, properties, train_mask, BOUND)

    @trans_builder.add_step(order=2)
    def log_pre_processing(dataset, properties, train_mask):
        utilities.log_pre_processing(dataset, properties, train_mask)

    @trans_builder.add_step(order=3)
    def categorical_conversion(
        dataset, properties, train_mask, categorical_levels=CATEGORICAL_LEV
    ):
        utilities.categorical_pre_processing(
            dataset, properties, train_mask, categorical_levels
        )

    @trans_builder.add_step(order=4)
    def bynary_label_conversion(dataset, properties, train_mask):
        utilities.bynary_label_conversion(dataset, properties, train_mask)

    proc.transformations = trans_builder.build()
    proc.apply()

    X_train, y_train = proc.get_train()
    X_vaild, y_valid = proc.get_valid()
    X_test, y_test = proc.get_test()

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_dataset = tabular_datasets.TabularDataset(
        X_train[prop.numeric_features], X_train[prop.categorical_features], device, y_train
    )
    valid_dataset = tabular_datasets.TabularDataset(
        X_vaild[prop.numeric_features], X_vaild[prop.categorical_features], device, y_valid
    )
    test_dataset = tabular_datasets.TabularDataset(
        X_test[prop.numeric_features], X_test[prop.categorical_features], device, y_test
    )

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEV):
        return utilities.one_hot_encoding(sample, categorical_levels)

    transformations = trans_builder.build()
    train_dataset.set_categorical_transformation(transformations)
    valid_dataset.set_categorical_transformation(transformations)
    test_dataset.set_categorical_transformation(transformations)

    BATCH_SIZE = 64
    WINDOW_SIZE = 256
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 10
    DROPUT = 0.1
    DIM_FF = 256
    LR = 0.001
    WHIGHT_DECAY = 0.01

    train_sampler = samplers.FairSlidingWindowSampler(
        train_dataset, y_train, 0, window_size=WINDOW_SIZE
    )
    valid_sampler = samplers.RandomSlidingWindowSampler(
        valid_dataset, window_size=WINDOW_SIZE
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
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
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

    inputs, _ = next(iter(train_dataloader))
    input_shape = inputs.shape[-1]

    input_encoding = input_encoder.InputEncoder(input_shape, EMBED_DIM)
    transformer_block = transformer.TransformerEncoderOnly(
        EMBED_DIM, NUM_HEADS, NUM_LAYERS, DIM_FF, DROPUT
    )
    class_head = classification_head.ClassificationHead(EMBED_DIM, 1)

    model = nn_classifier.NNClassifier(
        input_encoding, transformer_block, class_head
    ).to(device=device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of parameters: {total_params}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WHIGHT_DECAY,
    )

    N_EPOCH = 4
    EPOCH_STEPS = 64
    EPOCH_UNTIL_VALIDATION = 4
    PATIENCE = 2
    DELTA = 0.01

    train = trainer.Trainer(model, criterion, optimizer, device)
    # train.load_model()

    train.train(
        n_epoch=N_EPOCH,
        train_data_loader=train_dataloader,
        epoch_steps=EPOCH_STEPS,
        epochs_until_validation=EPOCH_UNTIL_VALIDATION,
        valid_data_loader=valid_dataloader,
        patience=PATIENCE,
        delta=DELTA,
    )

    metric = metrics.BinaryClassificationMetric()

    train.test(test_dataloader, metric)
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
