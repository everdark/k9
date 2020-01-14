"""Explore PyMC3: A probabilistic programming Python package.

Several things to know about PyMC3:
    1. It implements NUTS as the default sampler (same as Stan)
    2. Theano is the computing backend
    3. Interface is purely in Python; model code is compiled to C
For more information, please refer to the official document:
    https://docs.pymc.io/
"""

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats


np.random.seed(777)


# Test the API.
with pm.Model() as test_model:
    # Define a free parameter. (A prior.)
    y = pm.Normal("y", mu=0, sigma=1)

# Compute log-likelihood.
print(test_model.logp({"y": 0}))

# It should be equivalent to:
def dnorm(theta, mean=0, sd=1):
    """Probability density function of a Normal distribution."""
    return np.exp(-pow(theta - mean, 2) / (2*pow(sd, 2))) / np.sqrt(2*np.pi*pow(sd, 2))


print(np.log(dnorm(0)))


# --------------------------------------------------- #
# Binomial model. (De-generated logistic regression.) #
# --------------------------------------------------- #
# We use the example of analytical Bayesian discussed here:
# https://everdark.github.io/k9/notebooks/stats/bayesian/bayesian_modeling_explained.nb.html#122_analytical_bayesian
# Out of 10 coin flips, 6 are heads.
N = 10
y = 6
binom_model = pm.Model()
with binom_model:
    # Priors.
    theta = pm.Uniform("theta", lower=0, upper=1)

    # Likelihood of observations.
    yobs = pm.Binomial("yobs", n=N, p=theta, observed=y)


# Do MCMC sampling.
with binom_model:
    trace = pm.sample(500, chains=4)


# Summary table of the Bayesian estimates.
print(pm.summary(trace).round(3))


# --------------- #
# Logistic model. #
# --------------- #
# Use the example dataset we've explored in the notebook:
# https://everdark.github.io/k9/notebooks/stats/bayesian/bayesian_modeling_explained.nb.html#4_bayesian_logistic_regression_using_r
infile = "data/logit.csv"
data = pd.read_csv(infile)
data["b"] = 1  # Add a constant term.
X = data[["b", "x1", "x2", "x3", "x4"]].values
y = data["y"].values

# Build the model context.
logit_model = pm.Model()
with logit_model:
    # Priors.
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])

    # Model output.
    yhat = pm.math.sigmoid(pm.math.dot(X, beta))

    # Likelihood of observations.
    yobs = pm.Bernoulli("yobs", p=yhat, observed=y)


# Do MCMC sampling.
with logit_model:
    trace = pm.sample(500, chains=4)

print(trace["beta"].shape)
print(trace["beta"])

# Summary table of the Bayesian estimates.
print(pm.summary(trace).round(3))

# Trace plot. (Need package `arviz`: pip install arviz.)
pm.traceplot(trace)


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

# Build the model context.
linear_model = pm.Model()
with linear_model:
    # Priors.
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)  # Truncated Normal with only non-negative values.

    # Model output.
    yhat = pm.math.dot(X, beta)

    # Likelihood of observations.
    yobs = pm.Normal("yobs", mu=yhat, sigma=sigma, observed=y)


# Do MCMC sampling.
with linear_model:
    trace = pm.sample(500, chains=4)

print(trace["beta"].shape)
print(trace["beta"])

# Summary table of the Bayesian estimates.
print(pm.summary(trace).round(3))

# Trace plot. (Need package `arviz`: pip install arviz.)
pm.traceplot(trace)

