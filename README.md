# Self-Taught Data Science Playground

The repository is a collection of my self-taught notebooks for data science theories and practices.
A notebook is written in either [Jupyter](https://jupyter.org/) or [R markdown](https://rmarkdown.rstudio.com/).
The major programming languages used for most of the notebooks are [Python](https://www.python.org/) and/or [R](https://www.r-project.org/).
You may find me sometimes interoperate the two langauges in a single notebook.

## Notebooks

+ Statistics
    + [Bayesian Modeling Explained](https://everdark.github.io/k9/bayesian/bayesian_modeling_explained.nb.html)
+ Machine Learning
    + [Neural Networks Fundamentals](https://everdark.github.io/k9/neural_nets/neural_networks_fundamentals.nb.html)
    + [Matrix Factorization for Recommender Systems](https://everdark.github.io/k9/matrix_factorization/matrix_factorization.nb.html)
    + [Introduction to Learning-to-Rank](https://everdark.github.io/k9/learning_to_rank/learning_to_rank.html)
+ Natural Language Understanding
    + [On Subword Units](https://everdark.github.io/k9/natural_language_understanding/subword_units/subword_units.nb.html)
    + [Contex-Free Word Embeddings]
    + [Contex-Aware Word Embeddings]
+ Programming
    + R
        + [Asynchronous Programming in R]
        + [High Performance Computing with Rcpp]
+ Projects
    + [YouTube-8M Multi-Label Video Classification](https://everdark.github.io/k9/projects/yt8m/yt8m.html)
    + [A General-Purpose Neural Ranking Model]

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
+ Machine Learning
    + Factorization Machines
    + Gradient Boosting Trees
    + Recurrent Neural Nets
    + Sequence-to-Sequence Models
    + GANs
    + Reinforcement Learning Basics
    + Model Explanation
+ Statistics
    + Linear and Logistic Models: Econometrics v.s. Machine Learning
    + Naive Bayes
    + Bootstrap Sampling
    + Bayesian Model Diagnostic
+ Tools/Programming
    + TensorFlow 2.0 Hands-On
    + MXNet Hands-On
    + RASA Chatbot Framework Hands-On
    + Asynchronous Programming in R
+ Programming
    + R
        + Production Quality Shiny App Development
+ Projects
    + Model Deployment with gRRC

### Site

+ Dockerize each notebook (for complete reproducibility and portability)?
+ Tidy up dependencies for each notebook
