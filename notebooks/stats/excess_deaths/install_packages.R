#!/usr/bin/env Rscript

required_pkgs <- c(
    "rmarkdown",
    "metathis",
    "data.table",
    "DT",
    "readODS",
    "ggplot2",
    "ggrepel",
    "ggthemes",
    "ggforce",
    "rgeos",
    "sf",
    "rnaturalearth",
    "rnaturalearthdata",
    "lubridate",
    "countrycode",
    "WDI",
    "agtboost"
)
install.packages(required_pkgs, repos="http://cran.rstudio.com")
