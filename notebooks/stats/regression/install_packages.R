#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "lmtest",
    "sandwich"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
