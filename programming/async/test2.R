library(future)

# test 1

plan(multiprocess)
z <- 10
f <- future({
  get("z", envir=environment(), inherits=TRUE)  # Error.
})
tryCatch(value(f), error=function(e) print(e))

z <- 10
f <- future({
  get("z", envir=environment(), inherits=TRUE)  # This now is working even BEFORE the eval to z.
  z  # This makes the auto-search work and export the variable from global.
})
tryCatch(value(f), error=function(e) print(e))

plan(sequential)
z <- 10
f <- future({
  get("z", envir=environment(), inherits=TRUE)
})
tryCatch(value(f), error=function(e) print(e))


# test 2


plan(multiprocess)
e <- new.env()  # A global mutable.
e$x <- 0

f <- future({
  print(capture.output(pryr::where("e")))  # The original.
  e$x <- 42  # `e` is automatically searched and accessable, but changed in a copy.
  print(capture.output(pryr::where("e")))  # Now a local copy.
})
invisible(value(f))

ls.str(e)  # The original copy is intact.

plan(sequential)
e <- new.env()  # A global mutable.
e$x <- 0

f <- future({
  print(capture.output(pryr::where("e")))
  e$x <- 42
  print(capture.output(pryr::where("e")))
})
invisible(value(f))

ls.str(e)  # The original copy has been modified!


# test 3


plan(sequential)
x <- 0
f <- future({
  x <<- 42
  y <- 0
})
invisible(value(f))
x


plan(multiprocess)
x <- 0
f <- future({
  x <<- 42
  y <- 0
})
invisible(value(f))
x

