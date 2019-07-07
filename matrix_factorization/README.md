# Matrix Factorization

## R Dependencies for Notebook Rendering

+ [`rmarkdown`](https://rmarkdown.rstudio.com/)
+ [`reticulate`](https://github.com/rstudio/reticulate): To enable Python code interoperation in R.
+ [`DiagrammeR`](https://github.com/rich-iannone/DiagrammeR): To draw NN diagram.


## Python Dependencies for Coding Examples

Run

```sh
pip install -r requirements.txt
```

to install all the dependencies.
The code is only tested with Python 3.


## Render the Notebook

To render the notebook html output, run

```sh
Rscript render.R
```

One may need to setup the environment variable `PYTHON_PATH` to locate the python executable for `reticulate`.
For example:

```sh
PYTHON_PATH=/usr/bin/python3 Rscript render.R
```

The path can also be a `virtualenv` or a `condaenv` path.
