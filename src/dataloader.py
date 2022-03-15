from typing import List
from donkeycar.pipeline.types import TubDataset, TubRecord
import os
import unittest
import itertools
from collections import defaultdict
import numpy as np


def load_data(cfg, model, data_path='data', direction=None, size=None):
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

    tub_records_80_speed = {tn: load_records(tn, model, cfg, data_path, size) for tn in tub_names_80_speed}
    tub_records_85_speed = {tn: load_records(tn, model, cfg, data_path, size) for tn in tub_names_85_speed}
    tub_records_90_speed = {tn: load_records(tn, model, cfg, data_path, size) for tn in tub_names_90_speed}

    return tub_records_80_speed, tub_records_85_speed, tub_records_90_speed


def load_records(tub_name, model, cfg, data_path, size=None):
    dataset = TubDataset(
        config=cfg,
        tub_paths=[os.path.expanduser(os.path.join(data_path, tub_name))],
        seq_size=model.seq_size())

    records = dataset.get_records()[:size]
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

