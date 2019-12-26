# Demystify Modern Gradient Boosting Trees

## Notebook

[Link](https://everdark.github.io/k9/notebooks/ml/gradient_boosting/gbt.nb.html)

Modern gradient boosting trees (GBT) is undoubtedly one of the most powerful machine learning algorithms for traditional supervised learning tasks in the recent decade.
In this notebook we try to unbox two such powerful GBT frameworks: `xgboost` and `lightgbm`.
We will focus more on the methodology rather than their APIs to deeply understand how these algorithms work and why they are so effective comparing to the vanilla version of GBT,
which has been formally introduced nearly 20 years ago.

### Dependencies

To install the required R packages run:

```sh
Rscript install_packages.R
```

`lightgbm` is not available on CRAN as of 2019-12-26.
To install its R wrapper please refer to the [official instruction](https://github.com/microsoft/LightGBM/tree/master/R-package).

### Reproducibility

Pull the code base:

```sh
git clone git@github.com:everdark/k9.git
cd notebooks/ml/gradient_boosting
```

Then to render the html output:

```sh
Rscript render.R
```

