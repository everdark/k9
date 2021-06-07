# A Gentle Walkthrough of the Economist's Excess Deaths Model for COVID-19

## Notebook

[Link](https://everdark.github.io/k9/notebooks/stats/excess_deaths/excess_deaths.nb.html)

### Dependencies

We use [`renv`](https://github.com/rstudio/renv) to maintain R and its package version.

CAVEAT: One may need to install extra system dependencies related to `units` and `sf` packages.

### Reproducibility

Pull the code base:

```sh
git clone git@github.com:everdark/k9.git
cd notebooks/stats/excess_deaths
```

Restore packages to local:

```sh
Rscript -e "renv::restore()"
```

Then to render the html output:

```sh
Rscript render.R
```

