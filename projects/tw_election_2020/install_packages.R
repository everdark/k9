#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "magrittr",
    "data.table",
    "scales",
    "ggplot2",
    "ggrepel",
    "plotly",
    "sf",
    "lmtest",
    "sandwich"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
