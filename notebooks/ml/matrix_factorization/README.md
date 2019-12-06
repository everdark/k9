# Matrix Factorization

[Link](https://everdark.github.io/k9/notebooks/ml/matrix_factorization/matrix_factorization.nb.html)

## R Dependencies for Notebook Rendering

+ [`rmarkdown`](https://rmarkdown.rstudio.com/)
+ [`reticulate`](https://github.com/rstudio/reticulate): To enable Python code interoperation in R.
+ [`DiagrammeR`](https://github.com/rich-iannone/DiagrammeR): To draw NN diagram.
+ [`zoo`](https://cran.r-project.org/web/packages/zoo/index.html): To plot moving average series.

## Python Dependencies for Coding Examples

Run

```sh
pip install -r requirements.txt
```

to install all the dependencies.
The code is only tested with Python 3.

### Install `lightfm` on macOS with `openmp` Support

By default `lightfm` will be single-threaded under macOS.
To install it with `openmp` support, we need a homebrew `gcc` as the compiler and install from source.

```
# Install brew version gcc compiler if you don't have it already.
brew install gcc

# Clone the repo.
git clone git@github.com:lyst/lightfm.git
```

Then edit `setup.py` to have:

```py
use_openmp = True
```

Now install the package from local:

```sh
env CC=/usr/local/bin/g++-9 pip install -e .
```

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
