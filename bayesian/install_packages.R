#!/usr/bin/env Rscript

requirement_file <- "requirements.txt"
install.packages(readLines(requirement_file))
