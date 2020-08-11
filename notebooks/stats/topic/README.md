# Introduction to Topic Models

## Notebook

[Link](https://everdark.github.io/k9/notebooks/stats/topic/topic.nb.html)

A technical introduction along with working demo codes to two popular topic modeling approaches:
Latent Dirichlet Allocation (LDA) and Biterm Topic Modeling (BTM).
We use a very large collection of arXiv scholarly paper abstract (in English) for the former model and a very small collection of short text (in Traditional Chinese) about environmental violation fine for the latter.

### Dependencies

To install the required R packages run:

```sh
Rscript install_packages.R
```

And for Python:

```sh
pip install requests ckiptagger
```

### Reproducibility

Pull the code base:

```sh
git clone git@github.com:everdark/k9.git
cd notebooks/stats/topic
```

Then to render the html output:

```sh
PYTHON_PATH=$(which python) Rscript render.R
```

