"""Explore PyStan.

Stan is the state-of-the-art probablistic programming language written in C++.
PyStan is a Python wrapper for Stan.
For more information, please refer to the official document:
    https://mc-stan.org/users/interfaces/pystan
"""

import pystan
import pandas as pd
import numpy as np


# --------------------------------------------------- #
# Binomial model. (De-generated logistic regression.) #
# --------------------------------------------------- #
# We use the example of analytical Bayesian discussed here:
# https://everdark.github.io/k9/bayesian/bayesian_modeling_explained.nb.html#122_analytical_bayesian
# Out of 10 coin flips, 6 are heads.

# Create model.
binom_model_code = """
data {
    int y;
}

parameters {
    real<lower=0, upper=1> theta;
}

model {
    theta ~ uniform(0, 1);
    y ~ binomial(10, theta);
}
"""
binom_model = pystan.StanModel(model_code=binom_model_code)


# MCMC sampling.
binom_fit = binom_model.sampling(data={"y": 6}, iter=500, chains=4)

binom_fit.plot()
print(binom_fit)


# --------------- #
# Logistic model. #
# --------------- #
# Use the example dataset we've explored in the notebook:
# https://everdark.github.io/k9/bayesian/bayesian_modeling_explained.nb.html#4_bayesian_logistic_regression_using_r
infile = "data/logit.csv"
data = pd.read_csv(infile)
data["b"] = 1  # Add a constant term.
X = data[["b", "x1", "x2", "x3", "x4"]].values
y = data["y"].values


# Create model.
logit_model_code = """
data {
    int n;
    matrix[n, 5] X;
    int y[n];
}

parameters {
    vector[5] beta;
}

model {
    beta ~ normal(0, 1);
    y ~ bernoulli_logit(X*beta);
}
"""
logit_model = pystan.StanModel(model_code=logit_model_code)
data = {"n": len(y), "X": X, "y": y}


# MCMC sampling.
logit_fit = logit_model.sampling(data=data, iter=500, chains=4)

logit_fit.plot()
print(logit_fit)


# ------------- #
# Linear model. #
# ------------- #
# The error term distribution must be fully specified in order to make the model a probability one.
# Let's generate some simulated data.
sigma = 1
beta = [6, 4]
size = 100

X = np.random.normal(size=(size, 1))
X = np.hstack([np.ones_like(X), X])  # Include a constant term.
e = np.random.normal(size=size, loc=0, scale=sigma)
y = X.dot(beta) + e


# Create model.
linear_model_code = """
data {
    // Define sample size as intefer.
    // Define X as matrix.
    // Define y as real vector.
}

parameters {
    // Define regression coefficients as a vector,
    // Define the standard deviation of error.
}

model {
    // Specify prior for regressors.
    // Specify prior for the error term s.d..
    // Specify the model distribution for y = Xbeta.
}
"""
linear_model = pystan.StanModel(model_code=linear_model_code)
data = {"n": len(y), "X": X, "y": y}


# MCMC sampling.
linear_fit = linear_model.sampling(data=data, iter=500, chains=4)

linear_fit.plot()
print(linear_fit)

