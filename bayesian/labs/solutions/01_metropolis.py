"""Simple implementation of Metropolis-Hastings sampler for a Normal distribution.

Metropolis sampler is designed to draw random sample when only the unnormalized density
is available for the target distribution. Here we instead use a Normal distribution as
the target distribution for educational purpose.
"""

from functools import partial

import numpy as np
from scipy import stats
from plotly.offline import plot
import plotly.graph_objs as go


# Likelihood function for a Normal distribution.
# In real Bayesian inference we will use P(y|theta)P(theta) instead.
def dnorm(theta, mean=0, sd=1):
    """Probability density function for Normal distribution.

    dnorm(0) is equivalent to:
    ```
    from scipy import stats
    stats.norm.pdf(0, loc=0, scale=1)
    ```
    """
    return np.exp(-pow(theta - mean, 2) / (2*pow(sd, 2))) / np.sqrt(2*np.pi*pow(sd, 2))


class MetropolisSampler():

    def __init__(self, lik_fun, init_val=0):
        self.lik_fun = lik_fun
        self.current_val = init_val
        self.current_val_lik = self.lik_fun(self.current_val)
        self.n_accepted = 1
        self.accepted_samples = []

    def sample(self, size, scale=1):
        for _ in range(size):
            # The proposal distribution is N(0, 1).
            proposed_val = self.current_val + scale * stats.norm.rvs()
            proposed_val_lik = self.lik_fun(proposed_val)
            accept_prob = proposed_val_lik / self.current_val_lik
            if accept_prob > stats.uniform.rvs():
                self.current_val = proposed_val
                self.current_val_lik = proposed_val_lik
                self.n_accepted += 1
            # Note that if the proposed sample is rejected, the current sample is duplicated.
            self.accepted_samples.append(self.current_val)


# Try Metropolis sampling for a N(10, 5) target.
metropolis = MetropolisSampler(partial(dnorm, mean=10, sd=5), init_val=9)

samp_size = 5000
metropolis.sample(samp_size)
t = metropolis.accepted_samples

print("Accept Rate: ", metropolis.n_accepted / len(t))


# Trace plot of MCMC samples.
plot([go.Scatter(y=t)])


# Histogram of MCMC samples.
xticks = np.linspace(min(t), max(t), num=5000)
plot([
    go.Histogram(x=t, histnorm="probability density"),
    go.Scatter(x=xticks, y=stats.norm.pdf(xticks, 10, 5), name="Normal(10, 5)")
])


# Try a larger number of iteration.
metropolis.sample(5000)  # Continue with the last proposed sample.
t = metropolis.accepted_samples
xticks = np.linspace(min(t), max(t), num=5000)
plot([
    go.Histogram(x=t, histnorm="probability density"),
    go.Scatter(x=xticks, y=stats.norm.pdf(xticks, 10, 5), name="Normal(10, 5)")
])


# Try a larger number of iteration and drop the initial half samples. (Burn-in.)


# Try a different inital value.


# Now try MCMC sampling for another target distribution following Beta(2, 5).
# Can the sampler approximate correctly the target distribution?
metrop_beta = MetropolisSampler(partial(stats.beta.pdf, a=2, b=5), init_val=2 / (2 + 5))
metrop_beta.sample(40000, scale=.5)
rbeta = metrop_beta.accepted_samples[20000:]
xticks = np.linspace(min(rbeta), max(rbeta), num=5000)
plot([
    go.Histogram(x=rbeta, histnorm="probability density"),
    go.Scatter(x=xticks, y=stats.beta.pdf(xticks, 2, 5), name="Beta(2, 5)")
])


# Now try MCMC sampling for another target distribution following Uniform(0, 1).
# Can the sampler approximate correctly the target distribution?
metrop_unif = MetropolisSampler(stats.uniform.pdf, init_val=0)
metrop_unif.sample(40000)
runif = metrop_unif.accepted_samples[20000:]
xticks = np.linspace(min(runif), max(runif), num=1000)
plot([
    go.Histogram(x=runif, histnorm="probability density"),
    go.Scatter(x=xticks, y=stats.uniform.pdf(xticks), name="Uniform(0, 1)")
])

# Try different proposal scale to see how it affect the approximation.
# A very small scale? (For example scale = .01)
metrop_unif = MetropolisSampler(stats.uniform.pdf, init_val=0)
metrop_unif.sample(40000, scale=.01)
runif = metrop_unif.accepted_samples[20000:]
xticks = np.linspace(min(runif), max(runif), num=1000)
plot([
    go.Histogram(x=runif, histnorm="probability density"),
    go.Scatter(x=xticks, y=stats.uniform.pdf(xticks), name="Uniform(0, 1)")
])


# In general, our vanilla version of MCMC can approximiate the target distribution,
# but not as good as we would expect.
# This is because the implementation is barely minimum.
# A more serious implementation, for example,
# will use a mini-batch approach taking advantage of the Central Limit Theorem
# to further increase the quality of approxmiation.
