# k9-data-science

The source code to render my [website/blog](https://everdark.github.io/k9/).
It is based on [`distill`](https://github.com/rstudio/distill).

## Usage

To render the site:

```bash
make site  # or simply make
```

To create a new blog post:

```bash
make post-draft title="A New Post!"
```

To render the html of a post (from `.Rmd`):

```bash
make post path="./_posts/2022-04-12-a-new-post/a-new-post.Rmd"
```

## Depenencies

[`renv`](https://github.com/rstudio/renv/) is used to lock the dependencies for building the site.
Notebooks will have their own dependencies and are excluded from this site-wise lock state.
