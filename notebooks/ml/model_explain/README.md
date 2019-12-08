# On Model Explainability: From LIME, SHAP, to Explainable Boosting

## Notebook

[Link](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html)

Model explainability has gained more and more attention recently among machine learning practitioners.
Especially with the popularization of deep learning frameworks,
which further promotes the use of increasingly complicated models to improve accuracy.
In the reality, however, model with the highest accuracy may not be the one that can be deployed.
Trust is one important factor affecting the adoption of complicated models.
In this notebook we give a brief introduction to several popular methods on model explainability.
And we focus more on the hands-on which demonstrates how we can actually explain a model,
under a variety of use cases.

### Dependencies

To install all the Python packages used in the notebook, run:

```sh
pip install requirements.txt
```

Additionally for export static visualization on `interpret`,
we will need [`orca`](https://github.com/plotly/orca):

```sh
npm install -g electron@1.8.4 orca
```

[`npm`](https://www.npmjs.com/get-npm) is required.
For other installation method please refer to the official document of `orca`.

### Reproducibility

Pull the code base:

```sh
git clone git@github.com:everdark/k9.git
cd model_explain
```

Though most of the coding examples is written in Python,
the html notebook itself is written by R in `R Markdown`.
To install the required R packages run:

```sh
Rscript install_packages.R
```

Then to render the html output:

```sh
PYTHON_PATH=$(which python) Rscript render.R
```
