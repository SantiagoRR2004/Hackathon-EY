import pandas as pd
import numpy as np
import correlation
import random

nRows = 1000


for _ in range(100):

    # Random number of columns
    nColsX = random.randint(1, 20)
    nColsY = random.randint(1, 20)

    X = pd.Series([row for row in np.random.rand(nRows, nColsX)])
    Y = pd.Series([row for row in np.random.rand(nRows, nColsY)])

    sim = correlation.calculateCorrelation(X, Y)

    assert sim >= 0, "Correlation should be non-negative"
    assert sim <= 1, "Correlation should be at most 1"

    for _ in range(10):

        # Random permutation
        permX = np.random.permutation(nColsX)
        permY = np.random.permutation(nColsY)

        XPerm = pd.Series([row[permX] for row in X])
        YPerm = pd.Series([row[permY] for row in Y])

        simPerm = correlation.calculateCorrelation(XPerm, YPerm)

        assert np.isclose(
            sim, simPerm
        ), "Correlation should be invariant to column permutations"

        assert simPerm >= 0, "Correlation should be non-negative"
        assert simPerm <= 1, "Correlation should be at most 1"
