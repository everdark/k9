# Data Engineering Notebooks on Serverless Development

For the `rmarkdown` notebook generation we use [`renv`](https://github.com/rstudio/renv) to maintain the dependencies.

To re-generate the notebook:

```sh
git clone git@github.com:everdark/k9.git
cd notebooks/data_eng/serverless
Rscript -e "renv::restore()"
Rscript render.R
```
