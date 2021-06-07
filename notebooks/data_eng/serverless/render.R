#!/usr/bin/env Rscript

notebook_files <- c(
  "lambda_http_api.Rmd"
)

for ( nb in notebook_files ) {
  rmarkdown::render(nb, output_format="html_notebook")
}
