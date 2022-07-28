from typing import List
from donkeycar.pipeline.types import TubDataset, TubRecord
import os
import unittest
import itertools
from collections import defaultdict
import numpy as np


def load_data(cfg, data_path='data', direction=None, size=None):
    tub_names_80_speed = [
        '1-1-CC-80',
        '2-1-CW-80',
        '4-1-CC-80',
    ]
    tub_names_85_speed = [
        '1-3-CC-85',
        '2-3-CW-85',
        '3-3-CW-85',
        '4-3-CC-85'
    ]
    tub_names_90_speed = [
        '1-2-CC-90',
        '2-2-CW-90',
        '3-2-CW-90',
        '4-2-CC-90',
    ]

    if direction:
        tub_names_80_speed = [name for name in tub_names_80_speed if direction in name]
        tub_names_85_speed = [name for name in tub_names_85_speed if direction in name]
        tub_names_90_speed = [name for name in tub_names_90_speed if direction in name]

    tub_records_80_speed = {tn: load_records(tn, cfg, data_path, size) for tn in tub_names_80_speed}
    tub_records_85_speed = {tn: load_records(tn, cfg, data_path, size) for tn in tub_names_85_speed}
    tub_records_90_speed = {tn: load_records(tn, cfg, data_path, size) for tn in tub_names_90_speed}

    return tub_records_80_speed, tub_records_85_speed, tub_records_90_speed


def load_records(tub_name, cfg, data_path, size=None):
    dataset = TubDataset(
        config=cfg,
        tub_paths=[os.path.expanduser(os.path.join(data_path, tub_name))],
        seq_size=cfg.SEQUENCE_LENGTH)

    records = dataset.get_records()[:size]
    return records


def load_every_second_record(tub_name, cfg, data_path, size=None):
    dataset = TubDataset(
        config=cfg,
        tub_paths=[os.path.expanduser(os.path.join(data_path, tub_name))],
        seq_size=cfg.SEQUENCE_LENGTH)

    records = dataset.get_every_second_record()[:size]
    return records


def split_to_chunks(records: List[TubRecord], chunk_size: int = 100) -> List[np.array]:
    chunks = []
    for i in range(0, len(records), chunk_size):
        chunks.append(records[i:i + chunk_size])
    return chunks


class TestChunks(unittest.TestCase):

    def test_create_chunks_1(self):
        n_records = 201
        records = list(range(n_records))
        chunk_size = 100

        actual = split_to_chunks(records, chunk_size=chunk_size)

        self.assertEqual(len(actual), 3)
        self.assertEqual(n_records, len(list(itertools.chain(*actual))))

        self.assertEqual(len(actual[0]), 100)
        self.assertEqual(len(actual[1]), 100)
        self.assertEqual(len(actual[2]), 1)

    def test_create_chunks_2(self):
        n_records = 10
        records = list(range(n_records))
        chunk_size = 2

        actual = split_to_chunks(records, chunk_size=chunk_size)

        self.assertEqual(actual[0], [0, 1])
        self.assertEqual(actual[1], [2, 3])
        self.assertEqual(actual[2], [4, 5])
        self.assertEqual(actual[3], [6, 7])
        self.assertEqual(actual[4], [8, 9])


res = unittest.main(argv=[''], verbosity=3, exit=False)


def get_folds(data, n_folds=5, chunk_size=100):
    chunks = split_to_chunks(data, chunk_size)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=False)

    train_folds = defaultdict(list)
    test_folds = defaultdict(list)

    for chunk in chunks:
        for fold, (train_index, test_index) in enumerate(kf.split(chunk), start=1):
            train_folds[fold].append([chunk[i] for i in train_index])
            test_folds[fold].append([chunk[i] for i in test_index])

    train_folds = {k: list(itertools.chain(*v)) for k, v in train_folds.items()}
    test_folds = {k: list(itertools.chain(*v)) for k, v in test_folds.items()}

    return train_folds, test_folds


def get_fold_indices(data, n_folds=5, chunk_size=107):
    chunks = split_to_chunks(list(range(len(data))), chunk_size)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=False)

    train_indices = defaultdict(list)
    test_indices = defaultdict(list)

    for chunk in chunks:
        for fold, (train_index, test_index) in enumerate(kf.split(chunk), start=1):
            train_indices[fold].append([chunk[i] for i in train_index])
            test_indices[fold].append([chunk[i] for i in test_index])

    train_folds = [list(itertools.chain(*v)) for k, v in train_indices.items()]
    test_folds = [list(itertools.chain(*v)) for k, v in test_indices.items()]

    for train, test in zip(train_folds, test_folds):
        yield train, test


class TestGetFoldIndices(unittest.TestCase):

    def test_number_of_folds(self):
        n_records = 12
        records = list(range(n_records))

        n_folds = 2
        actual = get_fold_indices(records, n_folds=n_folds, chunk_size=3)

        self.assertEqual(len([_ for _ in actual]), n_folds)

    def test_sum_of_folds(self):
        n_records = 19252
        records = list(range(n_records))
        for train, test in get_fold_indices(records):
            self.assertEqual(len(test) + len(train), n_records)

    def test_2(self):
        n_records = 19252
        records = list(range(n_records))

        chunk_size = 107
        n_folds = 5

        expected_first_consecutive_train_indices_of_first_fold = list(range(22, 107))
        expected_first_consecutive_test_indices_of_first_fold = list(range(0, 22))

        for train, test in get_fold_indices(records, n_folds=n_folds, chunk_size=chunk_size):
            self.assertEqual(len(train), 15294)
            self.assertEqual(len(test), 3958)
            self.assertEqual(train[:(int((chunk_size / n_folds)*4))], expected_first_consecutive_train_indices_of_first_fold)
            self.assertEqual(test[:int(chunk_size / n_folds + 1)], expected_first_consecutive_test_indices_of_first_fold)
            break





