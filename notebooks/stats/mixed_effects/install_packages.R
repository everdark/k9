#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "ggplot2",
    "data.table",
    "plm",
    "lme4"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
