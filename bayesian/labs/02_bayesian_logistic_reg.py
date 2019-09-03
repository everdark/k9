"""Simple implementation of logistic regression model with both MLE and Bayesian approach.

This is for educational purpose so we are going to implement stuff mostly from scratch.
For MLE approach we will implement a vanilla stochastic gradient descent solver.
For Bayesian approach we will implement a vanilla Metropolis-Hastings sampler.
We only cover the solver but not the model diagnostic.
"""

import numpy as np
import pandas as pd
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# Read the sample dataset.
infile = "data/logit.csv"
data = pd.read_csv(infile)

# The sample dataset has 4 features and 1 outcome label.
print(data.shape)
print(data.head())

# Add a constant term.
data["b"] = 1

# Xy notation.
X = data[["b", "x1", "x2", "x3", "x4"]].values
y = data["y"].values


# First, let's solve the model via MLE, FROM SCRATCH--the hard way.
# Remember that MLE is to maximize data likelihood,
# or equivalently, to minimize the cross entropy.
def sigmoid(t):
    """Numerically stable sigmoid."""
    return np.exp(-np.logaddexp(0, -t))


def logloss(X, y, b):
    """Cross entropy loss function."""
    logloss_pos = y * np.log(sigmoid(X.dot(b)))
    logloss_neg = (1 - y) * np.log(1 - sigmoid(X.dot(b)))
    return - (logloss_pos + logloss_neg).mean()


# We derive analytically the gradient to the loss function.
# For detailed derivation please refer to
# https://everdark.github.io/k9/neural_nets/neural_networks_fundamentals.nb.html#12_logistic_regression
def grad_fn(X, y, b):
    """Gradient to the logloss function."""
    return - (y - sigmoid(X.dot(b))).dot(X) / X.shape[0]


def cross_entropy_gd_optimize(X, y, lr=.01, n_epoch=10, batch_size=64):
    b = np.random.normal(size=X.shape[1])
    l = [logloss(X, y, b)]
    for _ in range(n_epoch):
        # Shuffle the dataset before each epoch.
        sid = np.random.permutation(X.shape[0])
        Xs = X[sid,:]
        ys = y[sid]
        i = 0
        n_step = int(np.ceil(X.shape[0] / batch_size))
        for _ in range(n_step):
            Xb = Xs[i:i+batch_size,:]
            yb = ys[i:i+batch_size]
            b -= lr * grad_fn(Xb, yb, b)
            l.append(logloss(Xb, yb, b))
            i += batch_size
    return b, l


mle_beta, log_loss = cross_entropy_gd_optimize(X, y, lr=.1, n_epoch=3000)
print(mle_beta)

# Use a high-level API--the easy way.
# `statsmodels`:
# import statsmodels.api as sm
# print(sm.Logit(y, X).fit().params)

# `scikit-learn`:
# from sklearn.linear_model import LogisticRegression
# print(LogisticRegression(C=1e16, fit_intercept=False, solver="liblinear").fit(X, y).coef_)


# Now move onto a Bayesian estimate.
# The most important part is to implement:
# P(beta|y) \proto P(y|beta)P(beta), the numerator of the Bayes' Law for the posterior P(beta|y).
# Or the unnormalized posterior log-likelihood function based on observation.
def create_unnormalized_posterior_fn(X, y):
    def fn(beta, mu_beta=0, sigma_beta=2):
        v = ... # Compute the logit.
        logp = ... # Logloss of positive examples. Be aware of numerical stability.
        logq = ... # Logloss of negative examples. Be aware of numerical stability.
        logl = ... # Sum of logloss: log(P(y|beta)).
        return ... # log(P(y|beta) X P(beta))
    return fn


posterior_fn = create_unnormalized_posterior_fn(X, y)
posterior_fn(mle_beta)  # The posterior of our initial value (from MLE).


# Simple implementation of Metropolis-Hastings sampler with multiple parameters.
class MetropolisSampler():

    def __init__(self, loglik_fun, init_val):
        self.nparam = len(init_val)
        self.loglik_fun = loglik_fun
        self.current_val = init_val
        self.current_val_lik = loglik_fun(init_val)
        self.n_accepted = 1
        self.accepted_samples = []

    def sample(self, size, scale=1):
        for _ in range(size):
            # The proposal distribution is N(0, 1).
            proposed_val = ... # Proposed sample x2 is based on x1 + scale * N(0, 1).
            proposed_val_lik = ... # Log-likelihood of the proposed sample.
            accept_prob = ... # Likelihood ratio but in logarithm.
            if accept_prob > np.random.uniform():
                self.current_val = proposed_val
                self.current_val_lik = proposed_val_lik
                self.n_accepted += 1
            # Note that if the proposed sample is rejected, the current sample is duplicated.
            self.accepted_samples.append(self.current_val)


samp_size = 5000
metropolis = MetropolisSampler(posterior_fn, init_val=mle_beta)
metropolis.sample(samp_size, scale=.5)
mcmc_beta = np.vstack(metropolis.accepted_samples)

# Check acceptance rate.
print(metropolis.n_accepted / samp_size)

# Trace plot.
fig = make_subplots(rows=3, cols=2)
fig.add_trace(go.Scatter(y=mcmc_beta[:,0], name="$\\beta_0$"), row=1, col=1)
fig.add_trace(go.Scatter(y=mcmc_beta[:,0], name="$\\beta_1$"), row=2, col=1)
fig.add_trace(go.Scatter(y=mcmc_beta[:,0], name="$\\beta_2$"), row=3, col=1)
fig.add_trace(go.Scatter(y=mcmc_beta[:,0], name="$\\beta_3$"), row=1, col=2)
fig.add_trace(go.Scatter(y=mcmc_beta[:,0], name="$\\beta_4$"), row=2, col=2)
plot(fig, include_mathjax="cdn")

# Bayesian point estimates.
print(mcmc_beta.mean(axis=0))
