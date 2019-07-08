# Self-Taught Data Science

The repository is a collection of self-taught notebooks for data science practices.
A notebook is written in either [Jupyter](https://jupyter.org/) or [R markdown](https://rmarkdown.rstudio.com/).

The major programming language used for most of the notebooks is either [Python](https://www.python.org/) or [R](https://www.r-project.org/).

## Notebooks

+ Statistics
    + [Bayesian Modeling Explained](https://everdark.github.io/k9/bayesian/bayesian_modeling_explained.nb.html)
+ Machine Learning
    + [Introduction to Learning-to-Rank](https://everdark.github.io/k9/learning_to_rank/learning_to_rank.html)
    + [Neural Networks Fundamentals](https://everdark.github.io/k9/neural_nets/neural_networks_fundamentals.nb.html)
    + Matrix Factorization (W.I.P.)

## [Optional] Setup Python Environment

To ensure reproducibility it is recommended to use [`pyenv`](https://github.com/pyenv/pyenv) along with [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) to control both Python and package version.

To use `virtualenv` with `reticulate` in Rmd,
the involved Python must be installed with shared library:

```sh
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.0
```

`pyenv` support only Linux and macOS.
For Windows user it is recommended to use [`conda`](https://github.com/conda/conda) instead.

## TODO

### Topics
+ Factorization Machines
+ Gradient Boosting Machines
+ TensorFlow 2.0 Hands-On
+ Basics about Reinforcement Learning
+ Deep Neural Nets
    + Sequence-to-Sequence Model
        + Neural Machine Translation
    + GAN
+ Chatbot

### Site
+ Dockerize each notebook (for complete reproducibility)?
