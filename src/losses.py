import abc
from collections import defaultdict
from typing import List, Tuple

from donkeycar.config import Config
from donkeycar.parts.keras import KerasPilot
from model_loader import load_model
from donkeycar.pipeline.types import TubRecord

from donkeycar.pipeline.training import BatchSequence
from sklearn.metrics import mean_squared_error
import multiprocessing
import numpy as np
import tensorflow as tf


class ModelResults:
    def __init__(self,
                 model_class,
                 models_path: str,
                 seq_length: int,
                 direction: str,
                 n_folds: int,
                 tub_records_80_speed: dict,
                 tub_records_85_speed: dict,
                 tub_records_90_speed: dict,
                 train_folds: dict,
                 test_folds: dict,
                 config: Config):

        self.model_class = model_class
        self.models_path = models_path
        self.seq_length = seq_length
        self.direction = direction
        self.n_folds = n_folds
        self.tub_records_80_speed = tub_records_80_speed
        self.tub_records_85_speed = tub_records_85_speed
        self.tub_records_90_speed = tub_records_90_speed
        self.train_folds = train_folds
        self.test_folds = test_folds
        self.config = config

        self.results = defaultdict(dict)

    def predict_results(self):
        for fold, train_records in self.train_folds.items():
            trained_model = load_model(self.models_path, self.model_class, self.seq_length, self.direction,
                                       self.n_folds, fold)
            test_records = self.test_folds[fold]

            results_80_speed, results_85_speed, results_90_speed, results_90_speed_test, results_90_speed_train = self.get_test_metrics(
                trained_model, fold, test_records, train_records
            )

            self.results[fold]['0.8'] = results_80_speed
            self.results[fold]['0.85'] = results_85_speed
            self.results[fold]['0.9'] = results_90_speed
            self.results[fold]['0.9 test'] = results_90_speed_test
            self.results[fold]['0.9 train'] = results_90_speed_train

    def get_test_metrics(self, model: KerasPilot, fold: int, records_90_speed_test: List[TubRecord],
                         records_90_speed_train: List[TubRecord]) -> Tuple[List, List, List, List, List]:

        results_90_speed = []
        results_85_speed = []
        results_80_speed = []

        print(f'Getting 90-speed  mse-s for fold {fold}')
        for tub_name, test_records in self.tub_records_90_speed.items():
            tub_results_holder = Results(tub_name, self.direction, self.config, fold, model, test_records)
            print(tub_results_holder)
            results_90_speed.append(tub_results_holder)

        results_90_speed_test = Results('test90', self.direction, self.config, fold, model, records_90_speed_test)
        print(results_90_speed_test)

        results_90_speed_train = Results('train90', self.direction, self.config, fold, model, records_90_speed_train, is_train=True)
        print(results_90_speed_train)

        print(f'Getting 85-speed mse-s for fold {fold}')
        for tub_name, test_records in self.tub_records_85_speed.items():
            tub_results_holder = Results(tub_name, self.direction, self.config, fold, model, test_records)
            print(tub_results_holder)
            results_85_speed.append(tub_results_holder)

        print(f'Getting 80-speed mse-s for fold {fold}')
        for tub_name, test_records in self.tub_records_80_speed.items():
            tub_results_holder = Results(tub_name, self.direction, self.config, fold, model, test_records)
            print(tub_results_holder)
            results_80_speed.append(tub_results_holder)

        return results_80_speed, results_85_speed, results_90_speed, [results_90_speed_test], [results_90_speed_train]


def get_ground_truth_and_preds(kl, cfg, test_records):
    ground_truth = get_ground_truth(kl, test_records)
    test_preds = get_predictions(kl, cfg, test_records)
    return ground_truth, test_preds


def mse(v1, v2):
    v1 = np.array(v1)
    v1 = v1.reshape(v1.shape[0])
    v2 = np.array(v2)
    v2 = v2.reshape(v2.shape[0])
    assert v1.shape == v2.shape, (v1.shape, v2.shape)
    return mean_squared_error(v1, v2)


def get_ground_truth(kl, test_records: List[TubRecord]):
    if kl.seq_size() > 0:
        print(f'seq size {kl.seq_size()}')
        ground_truth = [r[kl.seq_size() - 1].underlying['user/angle'] for r in test_records]
    else:
        ground_truth = [r.underlying['user/angle'] for r in test_records]

    print(f'len ground_truth {len(ground_truth)}')

    return ground_truth


def get_predictions(kl, cfg, test_records):
    pipe = get_pipe(kl, cfg, test_records)
    steps = len(pipe)
    dataset = get_dataset_from_pipe(pipe)

    test_preds = kl.interpreter.model.predict(
        dataset,
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
        steps=steps,
        verbose=1)

    if 'Linear' in str(kl):
        test_preds = np.array(test_preds)[0][:len(test_records)]
    else:
        test_preds = np.array(test_preds)[:len(test_records)].T[0]

    print(f'shape test_preds {test_preds.shape}')

    return test_preds


def get_dataset_from_pipe(pipe):
    tune = tf.data.experimental.AUTOTUNE
    dataset = pipe.create_tf_data().prefetch(tune)
    return dataset


def get_pipe(model: KerasPilot, config: Config, records: List[TubRecord]):
    pipe = BatchSequence(model, config, records, is_train=False)
    return pipe


class Results:
    def __init__(self, tub_name, direction, cfg, fold, model, test_records, is_train=False):
        self.name = tub_name
        self.direction = direction
        self.fold = fold
        self.is_train = is_train
        ground_truths, predictions = get_ground_truth_and_preds(model, cfg, test_records)
        assert len(ground_truths) == len(predictions)
        self.ground_truths = ground_truths
        self.predictions = predictions

    @property
    def speed(self):
        return float(self.name[-2:]) / 100

    def __len__(self):
        return len(self.predictions)

    def __str__(self):
        return f'{self.name} has {len(self)} predictions with an mse of {mse(self.ground_truths, self.predictions)}'

    def __repr__(self):
        return f'Name: {self.name}, fold: {self.fold}, size: {len(self)}'
