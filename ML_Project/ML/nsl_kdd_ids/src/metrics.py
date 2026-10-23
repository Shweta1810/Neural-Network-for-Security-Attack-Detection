import numpy as np


def multiclass_acc(y_true, y_pred):
    return float((y_true == y_pred).mean())


def binary_acc(y_true, y_pred, normal_idx):
    """Normal vs abnormal accuracy: any non-normal prediction counts as 'abnormal'."""
    return float(((y_true == normal_idx) == (y_pred == normal_idx)).mean())


def binary_confusion(y_true, y_pred, normal_idx):
    t_att, p_att = y_true != normal_idx, y_pred != normal_idx
    tp = int((t_att & p_att).sum()); tn = int((~t_att & ~p_att).sum())
    fp = int((~t_att & p_att).sum()); fn = int((t_att & ~p_att).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                false_alarm_rate=fp / (fp + tn) if fp + tn else 0.0)
