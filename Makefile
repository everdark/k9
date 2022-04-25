# include TZ=UTC to avoid systemd complaint when running in wsl

.PHONY: all site post-draft post

all: site

site:
	TZ=UTC Rscript build_site.R

title ?= temp-title
post-draft:
	TZ=UTC Rscript -e 'distill::create_post("$(title)")'

path ?= _posts/temp-dir/temp-title.Rmd
post:
	TZ=UTC Rscript -e 'rmarkdown::render("$(path)", encoding = "UTF-8")'
