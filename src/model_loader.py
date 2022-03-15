import os


def init_model(model_class, seq_length):
    if 'Linear' not in str(model_class):
        return model_class(seq_length=seq_length)
    else:
        return model_class()


def build_model_name(model: object, fold: int, n_folds: int, direction: str):
    return f'{str(model)}-{fold}.fold-of-{n_folds}-{direction}'


def load_model(models_path, model_class, seq_length, direction, n_folds, fold):
    model = init_model(model_class, seq_length=seq_length)
    model_name = build_model_name(model, fold, n_folds=n_folds, direction=direction)
    model_path = os.path.join(models_path, f'{model_name}.h5')
    model.load(model_path)
    return model
