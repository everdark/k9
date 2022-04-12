# include TZ=UTC to avoid systemd complaint when running in wsl

.PHONY: all site post-draft post

all: site

site:
	TZ=UTC Rscript build_site.R

post-draft:
	TZ=UTC Rscript -e 'distill::create_post("$(title)")'

post:
	TZ=UTC Rscript -e 'rmarkdown::render("$(path)", encoding = "UTF-8")'
