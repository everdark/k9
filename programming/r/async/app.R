#!/usr/bin/env Rscript
# A minimum asynchronous shiny app.
# Usage:
#   Rscript app.R [sequential|multiprocess]
# By default (without argument) the app is run in sequential mode.

library(shiny)
library(future)
library(promises)


exec_plan <- commandArgs(trailingOnly=TRUE)[1]
if ( is.na(exec_plan) ) exec_plan <- "sequential"

plan(exec_plan)


# Define frontend code.
ui <- fluidPage(

  titlePanel("Async Shiny App"),
  textOutput("time"),
  actionButton("do", "Do some heavy works."),
  verbatimTextOutput("out")

)


do_heavy_work <- function() {
  st <- Sys.time()

  Sys.sleep(5) # Or anything expensive here.

  et <- Sys.time()
  list(st=st, et=et)
}


# Define backend code.
server <- function(input, output, session) {

  output$time <- renderText({
    invalidateLater(1000, session)
    paste("The current time is", Sys.time())
  })

  observeEvent(input$do, {
    st <- Sys.time()  # This only record when the app starts process the input but NOT when the user hit the button.
    output$out <- renderText({
      future(do_heavy_work()) %...>% {
        paste(
          "Heavy work done!",
          sprintf("Started at %s", st),
          sprintf("Ended at %s", .$et),
          sprintf("Time used: %s", .$et - st),
          sep="\n"
        )
      }
    })
  })

}

# Launch the app.
shinyApp(ui=ui, server=server, options=list(port=8787))
