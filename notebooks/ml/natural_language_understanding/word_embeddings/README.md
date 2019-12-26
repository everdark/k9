# On Subword Units

## Notebook

[Link](https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/word_embeddings/word_embeddings.nb.html)

Word embeddings are the building blocks for neural network models that solve natural language understanding (NLU) tasks.
In this notebook we review in details several influential models that are designed to learn context-free word embeddings for other downstream machine learning application,
a.k.a. transfer learning.
We also exercise extensively to use TensorFlow to demonstrate how we can implement each of the models.
At the end,
we demonstrate in short several popular frameworks to handle pre-trained embeddings.

### Dependencies

To install all the Python packages used in the notebook, run:

```sh
pip install requirements.txt
```

### Reproducibility

Pull the code base:

```sh
git clone git@github.com:everdark/k9.git
cd notebooks/ml/natural_language_understanding/word_embeddings
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
