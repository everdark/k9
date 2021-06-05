# Self-Taught Data Science Playground

The repository is a collection of my self-taught notebooks for data science theories and practices.
A huge effort is made to strike a balance between methodology derivation (with math) and hands-on coding.
The target audience is data science practitioners (including myself) with hands-on experiences who are seeking for more in-depth understandings of machine learning algorithms and relevant statistics.

Here to visit the web site [Hello, Data Science!](https://everdark.github.io/k9/) hosting all the notebooks in nicely rendered HTML.

## Notebooks Summary

`notebooks/`

A notebook is written in either [Jupyter](https://jupyter.org/) or [R markdown](https://rmarkdown.rstudio.com/).
The major programming languages used for most of the notebooks are [Python](https://www.python.org/) and/or [R](https://www.r-project.org/).
You may find me sometimes inter-operate the two langauges in a single notebook.
This is achieved thanks to [`reticulate`](https://github.com/rstudio/reticulate).

+ Statistics
    + [Bayesian Modeling Explained](https://everdark.github.io/k9/notebooks/stats/bayesian/bayesian_modeling_explained.nb.html)
    + [Bootstrap Sampling 101](https://everdark.github.io/k9/notebooks/stats/bootstrap/bootstrap.nb.html)
    + [Introduction to Topic Models](https://everdark.github.io/k9/notebooks/stats/topic/topic.nb.html)
    + [A Gentle Walkthrough of the Economist's Excess Deaths Model for COVID-19](https://everdark.github.io/k9/notebooks/stats/excess_deaths/excess_deaths.nb.html)
+ Machine Learning
    + [Neural Networks Fundamentals](https://everdark.github.io/k9/notebooks/ml/neural_nets/neural_networks_fundamentals.nb.html)
    + [Matrix Factorization for Recommender Systems](https://everdark.github.io/k9/notebooks/ml/matrix_factorization/matrix_factorization.nb.html)
    + [Introduction to Learning-to-Rank](https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html)
    + [General-Purpose Ranking Models](https://everdark.github.io/k9/notebooks/ml/neural_ranking/neural_ranking.nb.html)
    + [On Model Explainability](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html)
    + [Demystify Modern Gradient Boosting Trees](https://everdark.github.io/k9/notebooks/ml/gradient_boosting/gbt.nb.html)
+ Natural Language Understanding
    + [On Subword Units](https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html)
    + [Contex-Free Word Embeddings]
    + [Contex-Aware Word Embeddings]
+ Data Engineering
    + Infrastructure-as-Code: A Terraform AWS Use Case
    + Serverless Deployment: AWS Lambda with HTTP API
+ Programming
    + R
        + [Asynchronous Programming in R](https://everdark.github.io/k9/notebooks/eng/programming/r/async/async_r.nb.html)
        + [High Performance Computing in R using Rcpp](https://everdark.github.io/k9/notebooks/eng/programming/r/rcpp/rcpp.nb.html)
+ Projects
    + [YouTube-8M Multi-Label Video Classification](https://everdark.github.io/k9/projects/yt8m/yt8m.html)
    + [Visualization for Taiwanese General Election 2020](https://everdark.github.io/k9/projects/tw_election_2020/tw_election_2020.nb.html)
    + [A Note on Model Evaluation for Imbalanced Data](https://everdark.github.io/k9/projects/imbalance_eval/imbalance_eval.html)
    + [A Short Ride on AWS DeepRacer 2020](https://everdark.github.io/k9/projects/deepracer_2020/deepracer_2020.html)
    + [A Simulation Exercise for Your PS5 Lottery Draws](https://everdark.github.io/k9/projects/ps5/ps5.nb.html)

## Laboratory Scripts

`labs/`

These are quick-and-dirty scripts to explore a variety of open source machine learning tools.
They may not be completed and can be messy to read.

## [Optional] Setup Python Environment

To ensure reproducibility it is recommended to use [`pyenv`](https://github.com/pyenv/pyenv) along with [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) to control both Python and package version.

`pyenv` support only Linux and macOS.
For Windows user it is recommended to use [`conda`](https://github.com/conda/conda) instead.

### Install Different Python Version

To use `virtualenv` with `reticulate` in Rmd,
the involved Python must be installed with shared library:

```sh
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.0
```

### Create `virtualenv`

Each notebook has different package dependencies.
Here is an example to create an environment specific for the notebook on model explainability:

```
cd notebooks/ml/model_explain
pyenv virtualenv 3.7.0 k9-model-explain
pyenv local k9-model-explain
pip install --upgrade pip
pip install -r requirements.txt
```

## TODO

### Topics
+ Machine Learning
    + Factorization Machines
    + Recurrent Neural Nets
    + Sequence-to-Sequence Models
    + GANs
    + Reinforcement Learning Basics
    + Approximated Nearest Neighbor
+ Statistics
    + Law of Large Numbers and Central Limit Theorem
    + On Linear Regression: Machine Learning vs Econometrics
    + Linear Mixed Effects Models
    + Naive Bayes
    + Bayesian Model Diagnostic
    + Bayesian Time Series Forecasting
+ Tools/Programming
    + PyTorch Hands-On
    + RASA Chatbot Framework Hands-On
+ Programming
    + R
        + Production Quality Shiny App Development
    + Python
        + Dash for Interactive Dashboarding
+ Projects
    + Model Deployment with gRRC

### Site

+ Dockerize each notebook (for complete reproducibility and portability)?
+ Tidy up dependencies for each notebook
