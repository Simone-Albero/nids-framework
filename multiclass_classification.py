import logging
import pickle

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

def multiclass_classification(epoch_steps):
    CONFIG_PATH = "configs/dataset_properties.ini"
    DATASET_NAME = "nf_unsw_nb15_v2_anonymous"
    # TRAIN_PATH = "datasets/NF-UNSW-NB15-V2/NF-UNSW-NB15-V2-Train.csv"
    TRAIN_PATH = "datasets/NF-UNSW-NB15-V2/NF-UNSW-NB15-V2-Balanced-Train.csv"
    # TEST_PATH = "datasets/NF-UNSW-NB15-V2/NF-UNSW-NB15-V2-Test.csv"
    TEST_PATH = "datasets/NF-UNSW-NB15-V2/NF-UNSW-NB15-V2-Balanced-Test.csv"

    CATEGORICAL_LEVEL = 32
    BOUND = 100000000

    BATCH_SIZE = 64
    WINDOW_SIZE = 15
    EMBED_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 6
    DROPOUT = 0.3
    FF_DIM = 512
    LR = 0.0002
    WEIGHT_DECAY = 0.0005

    N_EPOCH = 1
    EPOCH_STEPS = epoch_steps
    # EPOCH_UNTIL_VALIDATION = 100
    # PATIENCE = 2
    # DELTA = 0.01

    named_prop = properties.NamedDatasetProperties(CONFIG_PATH)
    prop = named_prop.get_properties(DATASET_NAME)

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    trans_builder = transformation_builder.TransformationBuilder()

    min_values, max_values = utilities.min_max_values(df_train, prop, BOUND)
    unique_values = utilities.unique_values(df_train, prop, CATEGORICAL_LEVEL)
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
            dataset, prop, unique_values, CATEGORICAL_LEVEL
        )

    @trans_builder.add_step(order=4)
    def multi_class_label_conversion(dataset):
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
    torch.manual_seed(13)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = tabular_datasets.TabularDataset(
        X_train[prop.numeric_features],
        X_train[prop.categorical_features],
        y_train,
        device,
        "multiclass"
    )

    test_dataset = tabular_datasets.TabularDataset(
        X_test[prop.numeric_features], 
        X_test[prop.categorical_features], 
        y_test,
        device,
        "multiclass"
    )

    @trans_builder.add_step(order=1)
    def categorical_one_hot(sample, categorical_levels=CATEGORICAL_LEVEL):
        return utilities.one_hot_encoding(sample, categorical_levels)

    transformations = trans_builder.build()
    train_dataset.set_categorical_transformation(transformations)
    test_dataset.set_categorical_transformation(transformations)

    # train_sampler = samplers.RandomSlidingWindowSampler(
    #     train_dataset, window_size=WINDOW_SIZE
    # )
    # test_sampler = samplers.RandomSlidingWindowSampler(
    #     test_dataset, window_size=WINDOW_SIZE
    # )

    train_sampler = samplers.GroupWindowSampler(
        train_dataset, WINDOW_SIZE, df_train, "IPV4_DST_ADDR"
    )
    test_sampler = samplers.GroupWindowSampler(
        test_dataset, WINDOW_SIZE, df_test, "IPV4_DST_ADDR"
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

    num_classes = len(class_mapping)

    encoder = transformer.TransformerEncoder(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        window_size=WINDOW_SIZE,
    ).to(device)
    encoder.load_model_weights("saves/self_supervised_multi_encoder.pt")

    pre_trained_encoder = encoder
    for i, layer in enumerate(pre_trained_encoder.encoder.layers):
        if i <= 2: 
            for param in layer.parameters():
                param.requires_grad = False

    model = transformer.TransformerClassifier(
        num_classes=num_classes,
        input_dim=input_dim,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        window_size=WINDOW_SIZE,
    ).to(device)
    model.encoder = pre_trained_encoder

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of parameters: {total_params}")

    class_proportions = y_train.value_counts(normalize=True).sort_index()
    class_weights = 1.0 / class_proportions
    normalized_weights = class_weights / class_weights.sum()
    logging.info(f"class_weights: {normalized_weights}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(normalized_weights.values, dtype=torch.float32, device=device))
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
    model.save_model_weights(f"saves/multiclass.pt")

    metric = metrics.MulticlassClassificationMetric(num_classes)
    train.test(test_dataloader, metric)

if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    multiclass_classification(140)
    
    # for i in range(100, 500, 50):
    #     multiclass_classification(i)
