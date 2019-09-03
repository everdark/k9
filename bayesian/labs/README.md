# Laboratory for Bayesian Inference

This is a supplementary Python materials for the data science notebook of [Bayesian Modeling Explained](https://everdark.github.io/k9/bayesian/bayesian_modeling_explained.nb.html),
which originally was written with the R language.
The lab quickly explores MCMC algorithm for Bayesian modeling in Python.

Tested only with Python 3.

## Dependencies

To prepare for dependencies:

```py
pip install -r requirements.txt
```

The example dataset is borrowed from the `mcmc` R package and cached under `data` directory.
One can use `create_data.R` to re-create the file.
(Provided that the relevant R environment is ready.)

## Contents

+ `01_metropolis.py`: Built-from scratch Metropolis sampler.
+ `02_bayesian_logistic_reg.py`: Built-from scratch MLE and MCMC estimation for a logistic regression model.
+ `03_pymc3_intro.py`: A quick walk-through of using [`PyMC3`](https://github.com/pymc-devs/pymc3) to do Bayesian modeling with state-of-the-art MCMC sampler.

