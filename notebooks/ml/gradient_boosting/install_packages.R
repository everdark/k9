#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "rpart.plot",
    "xgboost",
    "microbenchmark"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
