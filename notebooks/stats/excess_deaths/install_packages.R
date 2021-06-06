#!/usr/bin/env Rscript

required_pkgs <- c(
  "rmarkdown",
  "metathis",
  "magrittr",
  "data.table",
  "DT",
  "readODS",
  "readxl",
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
missing_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if ( length(missing_pkgs) ) install.packages(missing_pkgs, repos="http://cran.rstudio.com")
