import numpy as np

def create_unlabeled_split(X, y, labeled_ratio=0.1):

    n = len(y)
    labeled_size = int(n * labeled_ratio)

    y_semi = np.copy(y)

    y_semi[labeled_size:] = -1

    return X, y_semi