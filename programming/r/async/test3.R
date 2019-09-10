library(future)
library(promises)
plan(multiprocess)

gv <- NULL

f <- future(1 + 1)
p1 <- then(f, onFulfilled=function(v) v + 1)
p2 <- then(p1, onFulfilled=function(v) gv <<- v + 2)

Sys.sleep(3)

gv  # Actual value from the last promise.


plan(sequential)

gv <- NULL

f <- future(1 + 1)
p1 <- then(f, onFulfilled=function(v) v + 1)
p2 <- then(p1, onFulfilled=function(v) gv <<- v + 2)

Sys.sleep(3)

gv  # Actual value from the last promise.
