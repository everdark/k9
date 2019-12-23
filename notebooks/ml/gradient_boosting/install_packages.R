#!/usr/bin/env Rscript

required_pkgs <- c(
    "reticulate",
    "rmarkdown",
    "rpart.plot",
    "xgboost"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
