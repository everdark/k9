#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "data.table",
    "ggplot",
    "plotly",
    "sf"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
