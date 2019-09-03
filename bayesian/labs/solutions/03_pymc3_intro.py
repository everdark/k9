"""Explore PyMC3: A probabilistic programming Python package."""

import numpy as np
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
binom_model = pm.Model()
# TBC.


# --------------- #
# Logistic model. #
# --------------- #
logit_model = pm.Model()
# TBC.


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

# Build the Bayesian model context.
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
# The default sampler is NUTS with multiple chains (the same as in Stan).
with linear_model:
    trace = pm.sample(500, chains=4)

print(trace["beta"].shape)
print(trace["beta"])

# Summary table of the Bayesian estimates.
print(pm.summary(trace).round(3))

# Trace plot. (Need package `arviz`: pip install arviz.)
pm.traceplot(trace)
