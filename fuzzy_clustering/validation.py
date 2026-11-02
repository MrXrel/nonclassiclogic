"""Partition agreement, used only after clustering has finished."""
import numpy as np


def adjusted_rand_index(reference, predicted):
    a, b = np.asarray(reference), np.asarray(predicted)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 2:
        raise ValueError("Для ARI нужны два набора меток одинаковой длины, минимум два объекта.")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max()+1, bi.max()+1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    pairs = lambda x: float(np.sum(x*(x-1)//2))
    together, row_pairs, column_pairs = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    total = len(a)*(len(a)-1)/2
    expected = row_pairs*column_pairs/total
    maximum = (row_pairs+column_pairs)/2
    return 1.0 if maximum == expected else (together-expected)/(maximum-expected)
