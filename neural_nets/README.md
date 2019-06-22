# Neural Networks Fundamentals

The notebook is created by [`rmarkdown`](https://rmarkdown.rstudio.com/) with [`reticulate`](https://github.com/rstudio/reticulate) to enable Python code interoperation.

## Python Dependencies

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

One needs to setup the environment variable `PYTHON_PATH` to locate the python executable for `reticulate`.
For example:

```sh
PYTHON_PATH=/usr/bin/python3 Rscript render.R
```

## Plotly Output

[`Plotly`](https://github.com/plotly/plotly.py) is used for visualization task in this notebook.
To avoid over-sizing of each plot, `plotly.js` library is only included once in the notebook header.
This will result in RStudio not able to preview the graphical output interactively.

