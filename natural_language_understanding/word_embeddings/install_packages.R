#!/usr/bin/env Rscript

required_pkgs <- c(
    "reticulate",
    "ggplot2",
    "data.table",
    "rmarkdown",
    "text2vec"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
