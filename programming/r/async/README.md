# Asynchronous Programming in R

## Notebook

[Link](https://everdark.github.io/k9/programming/r/async/async_r.nb.html)

The notebook is a practical walk-through of implementing asynchronous programming in R using package `future` and `promises`,
mainly for the purpose of kick-starting developing a scalable `shiny` application.
Indeed, all 3 packages come with very well-structured official tutorials already.
This notebook serves more as a minimally sufficient one-stop reference for developers to quickly hands-on on the topic and get ready for the actual application development.

### Dependencies

Run

```sh
Rscript install_packages.R
```

to collect dependencies required to reproduce the notebook.

To re-create the html notebook:

```sh
git clone git@github.com:everdark/k9.git
cd programming/r/async
Rscript render.R
```
