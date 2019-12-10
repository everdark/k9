#!/usr/bin/env Rscript
# Build a sentiment classifier for movie review dataset
# with word embeddings trained from scratch.
# Note
# The R interface has better support for using keras as top-level module
# than as tensorflow submodule.

library(tensorflow)
library(keras)


# Fetch movie review dataset (already vectorized with word index).
imdb <- dataset_imdb()

# This is one example from training set:
print(imdb$train$x[[1]])

# Padding.
X_train <- pad_sequences(imdb$train$x, padding="post")
maxlen <- ncol(X_train)
X_test <- pad_sequences(imdb$test$x, padding="post", maxlen=maxlen)

# Get vocab size.
vocab_size <- max(max(X_train), max(X_test))

# Compile model.
inputs <- layer_input(shape=c(maxlen))
outputs <- inputs %>%
  layer_embedding(input_dim=vocab_size + 1, output_dim=embedding_size, mask_zero=TRUE) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(embedding_size / 2, activation="relu") %>%
  layer_dense(1, activation="sigmoid")
model <- keras_model(inputs=inputs, outputs=outputs)
model %>% compile(
  optimizer="adam",
  loss="binary_crossentropy",
  metrics=c("accuracy")
)

summary(model)

model_file <- "keras_imdb_model.h5"

# Train.
model %>% fit(
  x=X_train, y=imdb$train$y, epochs=10, batch_size=256,
  validation_data=list(X_test, imdb$test$y),
  validation_steps=20,
  callbacks=list(
    callback_early_stopping(monitor="val_loss", patience=2),
    callback_model_checkpoint(model_file, monitor="val_loss", save_best_only=TRUE)
  ),
  verbose=1
)

# Make predictions.
p <- predict(model, X_test)

# Evaluate.
metrics <- evaluate(model, X_test, imdb$test$y, verbose=0)
print(metrics)

# Check model save/load.
model2 <- load_model_hdf5(model_file)
metrics2 <- evaluate(model2, X_test, imdb$test$y, verbose=0)
print(metrics2)
