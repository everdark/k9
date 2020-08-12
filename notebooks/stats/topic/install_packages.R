#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "reticulate",
    "data.table",
    "jsonlite",
    "ggplot2",
    "text2vec",
    "BTM",
    "devtools",
    "rgl"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
devtools::install_github("gadenbuie/metathis")
devtools::install_github("cpsievert/LDAvis")  # We want the dev version that can reserve topic index.
