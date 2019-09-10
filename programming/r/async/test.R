#!/usr/bin/env Rscript

library(future)
library(promises)
library(shiny)

plan(multiprocess)


# Define frontend code.
ui <- fluidPage(

  titlePanel("Test super-assignment in promise"),
  fluidRow(
    column(3, actionButton("do", "Do something.")),
    column(3, uiOutput("out"))
  )

)


# Define backend code.
server <- function(input, output, session) {
  # This works.
  gv <- NULL

  f <- reactive({
    future(1 + 1)
  })

  observeEvent(input$do, {

    p <- then(f(), onFulfilled=function(v) gv <<- v + 1)

    output$out <- renderText({
      Sys.sleep(3)
      gv
    })
  })

}


# Launch the app.
shinyApp(ui=ui, server=server, options=list(port=8787))



test <- function() {
  print(environment())
  print(parent.frame())
  # This doesn't work.
  gv <- NULL
  f <- future(1 + 1)
  g <- function() {
    print("g:")
    print(environment())
    print("g parent:")
    print(parent.frame())
    p <- then(f, onFulfilled=function(v) {
      print("p:")
      print(environment())
      print("p parent:")
      print(parent.frame())  # The parent is NOT g.
      gv <<- v + 1
    })
    Sys.sleep(3)
    print(gv)
  }
  g()  # With or without g wrapper the result is the same.
  gv
}

test()


test_env <- function() {
  tt <- environment()
  gv <- NULL
  f <- future({
    print("f:")
    print(environment())
    print("f parent:")
    print(parent.frame())
    1 + 1
  })
  p <- then(f, onFulfilled=function(v) {
    print("p:")
    print(environment())
    print("p parent:")
    print(parent.frame())
    tt$gv <<- v + 1
  })
  Sys.sleep(3)
  print(gv)
  Sys.sleep(3)
  print(gv)
}


test_env()
