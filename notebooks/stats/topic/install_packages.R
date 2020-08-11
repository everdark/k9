#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "reticulate",
    "data.table",
    "jsonlite",
    "ggplot2",
    "text2vec",
    "BTM",
    "LDAvis",
    "rgl"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
