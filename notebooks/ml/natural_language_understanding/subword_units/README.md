# On Subword Units

## Notebook

[Link](https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html)

Langauge segmentation or tokenization is the very first step toward a natural language understanding (NLU) task.
The state-of-the-art NLU model usually involves neural networks with word embeddings as the workhorse to encode raw text onto vector space.
A fixed vocabulary is pre-determined in order to facilitate such setup.
With the challenge of out-of-vocabulary issue and non-whitespace-delimited language,
we use subword units to further decompose raw texts into substrings.
In this notebook we summarize the technique of subword segmentation in details with Python coding examples.
We also provide a general usage walk-through for Google's open source library *SentencePiece*,
a powerful language-agnostic unsupervised subword segmentation tool.

### Dependencies

To install all the Python packages used in the notebook, run:

```sh
pip install requirements.txt
```

### Reproducibility

Pull the code base:

```sh
git clone git@github.com:everdark/k9.git
cd natural_language_understanding/subword_units
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
