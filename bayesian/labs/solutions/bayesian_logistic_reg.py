"""Simple implementation of Bayesian logistic regression model."""

import numpy as np
import pandas as pd


# Read the sample dataset.
infile = "data/logit.csv"
data = pd.read_csv(infile)

# The sample dataset has 4 features and 1 outcome label.
data.head()

# Add a constant term.
data["b"] = 1

# Xy notation.
X = data[["b", "x1", "x2", "x3", "x4"]].values
y = data["y"].values

# Initial values.
beta = np.array([0, 0, 0, 0, 0])
beta = np.array([1, 1, 1, 1, 1])


# Unnormalized log posterior likelihood function based on observation.
def create_unnormalized_posterior_fn(X, y):
    def fn(beta, mu_beta=0, sigma_beta=2):
        v = X.dot(beta)
        logp = np.where(v < 0, v - np.log1p(np.exp(v)), - np.log1p(np.exp(-v)))
        logq = np.where(v < 0, - np.log1p(np.exp(v)), - v - np.log1p(np.exp(-v)))
        logl = logp[y == 1].sum() + logq[y == 0].sum()
        return logl - pow(beta - mu_beta, 2).sum() / (2 * pow(sigma_beta, 2))
    return fn


posterior_fn = create_unnormalized_posterior_fn(X, y)


