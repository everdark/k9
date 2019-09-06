library(future)
library(promises)
library(shiny)

plan(multiprocess)

test <- function() {

    f <- future(1 + 1)
    p <- then(f, onFulfilled=function(v) gv <<- v + 1)

    Sys.sleep(5)
    print(gv)
}

test()



