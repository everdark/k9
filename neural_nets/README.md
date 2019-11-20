# Neural Networks Fundamentals

[Link](https://everdark.github.io/k9/neural_nets/neural_networks_fundamentals.nb.html)

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

## Plotly Output

[`Plotly`](https://github.com/plotly/plotly.py) is used for visualization task in this notebook.
To avoid over-sizing of each plot, `plotly.js` library is only included once in the notebook header.
This will result in RStudio not able to preview the graphical output interactively.

