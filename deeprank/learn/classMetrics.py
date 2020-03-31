import warnings

import numpy as np

# info
# https://en.wikipedia.org/wiki/Precision_and_recall


def sensitivity(yp, yt):
    """sensitivity, recall or true positive rate (TPR)

    Args:
        yp (array): predictions
        yt (array): targets

    Returns:
        float: sensitivity value
    """
    tp = true_positive(yp, yt)
    p = positive(yt)
    if p == 0:
        tpr = float('inf')
        warnings.warn(
            f'Number of positive cases is 0, '
            f'TPR or sensitivity is assigned as inf')
    else:
        tpr = tp / p
    return tpr


def specificity(yp, yt):
    """specificity, selectivity or true negative rate (TNR)

    Args:
        yp (array): predictions
        yt (array): targets

    Returns:
        float: specificity value
    """
    tn = true_negative(yp, yt)
    n = negative(yt)
    if n == 0:
        warnings.warn(
            f'Number of negative cases is 0, '
            f'TNR or sepcificity is assigned as inf')
        tnr = float('inf')
    else:
        tnr = tn / n
    return tnr


def precision(yp, yt):
    """precision or positive predictive value (PPV)

    Args:
        yp (array): predictions
        yt (array): targets

    Returns:
        float: precision value
    """
    tp = true_positive(yp, yt)
    fp = false_positive(yp, yt)
    if tp + fp == 0:
        warnings.warn(
            f'Total number of true positive and false positive cases is 0, '
            f'PPV or precision is assigned as inf')
        ppv = float('inf')
    else:
        ppv = tp / (tp + fp)
    return ppv


def accuracy(yp, yt):
    """Accuracy.

    Args:
        yp (array): predictions
        yt (array): targets

    Returns:
        float: accuracy value
    """
    tp = true_positive(yp, yt)
    tn = true_negative(yp, yt)
    p = positive(yt)
    n = negative(yt)
    acc = (tp + tn) / (p + n)
    return acc


def F1(yp, yt):
    """F1 score.

    Args:
        yp (array): predictions
        yt (array): targets

    Returns:
        float: F1 score
    """
    tp = true_positive(yp, yt)
    fp = false_positive(yp, yt)
    fn = false_negative(yp, yt)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


def true_positive(yp, yt):
    """number of true positive cases.

    Args:
        yp (array): predictions
        yt (array): targets
    """
    yp, yt = _to_bool(yp), _to_bool(yt)
    tp = np.logical_and(yp, yt)
    return(np.sum(tp))


def true_negative(yp, yt):
    """number of true negative cases.

    Args:
        yp (array): predictions
        yt (array): targets
    """
    yp, yt = _to_bool(yp), _to_bool(yt)
    tn = np.logical_and(yp == False, yt == False)
    return(np.sum(tn))


def false_positive(yp, yt):
    """number of false positive cases.

    Args:
        yp (array): predictions
        yt (array): targets
    """
    yp, yt = _to_bool(yp), _to_bool(yt)
    fp = np.logical_and(yp, yt == False)
    return(np.sum(fp))


def false_negative(yp, yt):
    """number of false false cases.

    Args:
        yp (array): predictions
        yt (array): targets
    """
    yp, yt = _to_bool(yp), _to_bool(yt)
    fn = np.logical_and(yp == False, yt == True)
    return(np.sum(fn))


def positive(yt):
    """The number of real positive cases.

    Args:
        yt (array): targets
    """
    yt = _to_bool(yt)
    return np.sum(yt)


def negative(yt):
    """The nunber of real negative cases.

    Args:
        yt (array): targets
    """
    yt = _to_bool(yt)
    return(np.sum(yt == False))


def _to_bool(x):
    """convert array values to boolean values.

    Args:
        x (array): values should be  0 or 1

    Returns:
        array: boolean array
    """
    return x.astype(bool)
