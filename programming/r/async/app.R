#!/usr/bin/env Rscript

library(shiny)
library(future)
library(promises)

plan(sequential)

# Define frontend code.
ui <- fluidPage(

  titlePanel("Async Shiny App"),
  fluidRow(
    column(6, actionButton("do", "Do some heavy works."))
  ),
  fluidRow(
    column(12, verbatimTextOutput("out"))
  )

)


do_heavy_work <- function() {
  st <- Sys.time()

  Sys.sleep(3) # Or anything expensive here.

  et <- Sys.time()
  list(st=st, et=et)
}


# Define backend code.
server <- function(input, output, session) {

  observeEvent(input$do, {
    # TODO:
    # Timing is not working as expected.
    # We need to record the user button hit time but if the app is busy in single-thread mode how can we do that?
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
