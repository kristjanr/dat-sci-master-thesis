from model_loader import build_model_name
from donkeycar.utils import train_test_split
from donkeycar.pipeline.training import BatchSequence
import tensorflow as tf


def train_model(model, models_path: str, fold: int, train_records: list, n_folds, direction, cfg: Config):
    model_name = build_model_name(model, fold, n_folds, direction)
    model_path = f'{models_path}{model_name}.h5'
    if cfg.WANDB_ENABLED:
        run = init_wandb(fold, model_name)

    train(model, model_path, cfg, train_records)

    if cfg.WANDB_ENABLED:
        run.finish()

    return model


def init_wandb(fold, model_name, model_class, n_folds, direction, cfg: Config):
    import wandb
    config = {
        "model_class_name": str(model_class),
        "total_folds": n_folds,
        "fold": fold,
        "model_name": model_name,
        "SEQUENCE_LENGTH": cfg.SEQUENCE_LENGTH,
        "EARLY_STOP_PATIENCE": cfg.EARLY_STOP_PATIENCE,
        "DIRECTION": direction,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "MAX_EPOCHS": cfg.MAX_EPOCHS,
    }
    return wandb.init(project="master-thesis", entity="kristjan", config=config)


def train(kl, model_path, cfg, data):
    dataset_train, dataset_validate, train_size, val_size = prep_fold_data(kl, cfg, data)
    history = kl.train(model_path=model_path,
                       train_data=dataset_train,
                       train_steps=train_size,
                       batch_size=cfg.BATCH_SIZE,
                       validation_data=dataset_validate,
                       validation_steps=val_size,
                       epochs=cfg.MAX_EPOCHS,
                       verbose=cfg.VERBOSE_TRAIN,
                       min_delta=cfg.MIN_DELTA,
                       patience=cfg.EARLY_STOP_PATIENCE,
                       show_plot=cfg.SHOW_PLOT,
                       add_wandb_callback=cfg.WANDB_ENABLED)

    return history


def prep_fold_data(kl, cfg, data):
    training_records, validation_records = train_test_split(data, shuffle=False,
                                                            test_size=(1. - cfg.TRAIN_TEST_SPLIT))
    print(f'Records # Training {len(training_records)}')
    print(f'Records # Validation {len(validation_records)}')
    # We need augmentation in validation when using crop / trapeze
    training_pipe = BatchSequence(kl, cfg, training_records, is_train=True)
    validation_pipe = BatchSequence(kl, cfg, validation_records, is_train=False)
    tune = tf.data.experimental.AUTOTUNE
    dataset_train = training_pipe.create_tf_data().prefetch(tune)
    dataset_validate = validation_pipe.create_tf_data().prefetch(tune)
    train_size = len(training_pipe)
    val_size = len(validation_pipe)
    assert val_size > 0, "Not enough validation data, decrease the batch size or add more data."
    return dataset_train, dataset_validate, train_size, val_size
