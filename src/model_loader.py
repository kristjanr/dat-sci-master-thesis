import os

from donkeycar.utils import get_model_by_type


def build_model_name(model: object, fold: int, n_folds: int, direction: str):
    return f'{str(model)}-{fold}.fold-of-{n_folds}-{direction}'


def load_model(models_path, model_type, config, direction, n_folds, fold):
    model = get_model_by_type(model_type, config)
    model_name = build_model_name(model, fold, n_folds=n_folds, direction=direction)
    model_path = os.path.join(models_path, f'{model_name}.h5')
    model.load(model_path)
    return model
