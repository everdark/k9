#!/usr/bin/env Rscript

required_pkgs <- c(
    "reticulate",
    "rmarkdown",
    "rpart.plot"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
