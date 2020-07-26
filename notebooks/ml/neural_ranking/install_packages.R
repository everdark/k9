#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "data.table",
    "sentencepiece",
    "tidytext",
    "ruimtehol"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
