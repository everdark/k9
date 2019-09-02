#!/usr/bin/env Rscript

library(mcmc)

data(logit)
write.csv(logit, "data/logit.csv", row.names=FALSE)

