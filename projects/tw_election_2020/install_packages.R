#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "data.table",
    "scales",
    "ggplot2",
    "ggrepel",
    "plotly",
    "RColorBrewer",
    "sf",
    "lmtest",
    "sandwich",
    "rvest"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
