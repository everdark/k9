# High Performance Computing in R using `Rcpp`

## Notebook

[Link](https://everdark.github.io/k9/programming/r/rcpp/rcpp.nb.html)

R is inherently a high-performance scripting language since its core is written in C.
Writing a high-performance R program,
however,
may not be as straightforward as it is for some other general purpose languages.
This notenook demonstrates quickly the key concept in writing a good R program and a fact check about the decomposition of the language source code.
It also demonstrates in more details with several working examples and benchmarks on how we can boost the performance when the underlying task is not easy to optimize in native R code:
by using Rcpp.

### Dependencies

Run

```sh
Rscript install_packages.R
```

to collect dependencies required to reproduce the notebook.

To re-create the html notebook:

```sh
git clone git@github.com:everdark/k9.git
cd programming/r/rcpp
Rscript render.R
```
