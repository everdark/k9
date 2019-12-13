#!/usr/bin/env Rscript

required_pkgs <- c(
    "reticulate",
    "rmarkdown",
    "xml2",
    "ggplot2"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
