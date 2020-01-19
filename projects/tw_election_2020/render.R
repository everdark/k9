#!/usr/bin/env Rscript

# Workaround weird issue causing rendering failure:
# https://github.com/rstudio/rmarkdown/issues/1762
# Overwrite the function causing the trouble (disable the sanity check).

extractPreserveChunks <- function(strval) {

  # Literal start/end marker text. Case sensitive.
  startmarker <- "<!--html_preserve-->"
  endmarker <- "<!--/html_preserve-->"
  # Start and end marker length MUST be different, it's how we tell them apart
  startmarker_len <- nchar(startmarker)
  endmarker_len <- nchar(endmarker)
  # Pattern must match both start and end markers
  pattern <- "<!--/?html_preserve-->"

  # It simplifies string handling greatly to collapse multiple char elements
  if (length(strval) != 1)
    strval <- paste(strval, collapse = "\n")

  # matches contains the index of all the start and end markers
  matches <- gregexpr(pattern, strval)[[1]]
  lengths <- attr(matches, "match.length", TRUE)

  # No markers? Just return.
  if (matches[[1]] == -1)
    return(list(value = strval, chunks = character(0)))

  # If TRUE, it's a start; if FALSE, it's an end
  boundary_type <- lengths == startmarker_len

  # Positive number means we're inside a region, zero means we just exited to
  # the top-level, negative number means error (an end without matching start).
  # For example:
  # boundary_type - TRUE TRUE FALSE FALSE TRUE FALSE
  # preserve_level - 1 2 1 0 1 0
  preserve_level <- cumsum(ifelse(boundary_type, 1, -1))

  # Sanity check.
  if (any(preserve_level < 0) || tail(preserve_level, 1) != 0) {
    #stop("Invalid nesting of html_preserve directives")
  }

  # Identify all the top-level boundary markers. We want to find all of the
  # elements of preserve_level whose value is 0 and preceding value is 1, or
  # whose value is 1 and preceding value is 0. Since we know that preserve_level
  # values can only go up or down by 1, we can simply shift preserve_level by
  # one element and add it to preserve_level; in the result, any value of 1 is a
  # match.
  is_top_level <- 1 == (preserve_level + c(0, preserve_level[-length(preserve_level)]))

  preserved <- character(0)

  top_level_matches <- matches[is_top_level]
  # Iterate backwards so string mutation doesn't screw up positions for future
  # iterations
  for (i in seq.int(length(top_level_matches) - 1, 1, by = -2)) {
    start_outer <- top_level_matches[[i]]
    start_inner <- start_outer + startmarker_len
    end_inner <- top_level_matches[[i+1]]
    end_outer <- end_inner + endmarker_len

    id <- htmltools:::withPrivateSeed(
      paste("preserve", paste(
        format(as.hexmode(sample(256, 8, replace = TRUE)-1), width=2),
        collapse = ""),
        sep = "")
    )

    preserved[id] <- gsub(pattern, "", substr(strval, start_inner, end_inner-1))

    strval <- paste(
      substr(strval, 1, start_outer - 1),
      id,
      substr(strval, end_outer, nchar(strval)),
      sep="")
    substr(strval, start_outer, end_outer-1) <- id
  }

  list(value = strval, chunks = preserved)
}

assignInNamespace("extractPreserveChunks", extractPreserveChunks, "htmltools")
rmarkdown::render("tw_election_2020.Rmd", output_format="html_notebook")
