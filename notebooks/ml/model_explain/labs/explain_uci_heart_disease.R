library(data.table)
library(ranger)
library(pROC)
library(iml)


set.seed(64)

# https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
data_dir <- "~/.keras/datasets"
infile <- file.path(data_dir, "processed.cleveland.data")

ucihd_attr = c(
  "age",
  "sex",      # 0 = female 1 = male
  "cp",       # chest pain type 1: typical angina 2: atypical angina 3: non-anginal pain 4: asymptomatic
  "trestbps", # resting blood pressure (in mm Hg on admission to the hospital)
  "chol",     # serum cholestoral in mg/dl
  "fbs",      # (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
  "restecg",  # resting electrocardiographic results 0: normal 1: having ST-T wave abnormality 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
  "thalach",  # maximum heart rate achieved
  "exang",    # exercise induced angina (1 = yes; 0 = no)
  "oldpeak",  # ST depression induced by exercise relative to rest
  "slope",    # the slope of the peak exercise ST segment
  "ca",       # number of major vessels (0-3) colored by flouroscopy
  "thal",     # 3 = normal; 6 = fixed defect; 7 = reversable defect
  "label"     # diagnosis of heart disease (angiographic disease status) 0: < 50% diameter narrowing 1-4: > 50% diameter narrowing
)

ucihd <- fread(infile, header=FALSE, col.names=ucihd_attr, na.strings="?")
ucihd[is.na(ca), ca:=-1]
ucihd[, label:=factor(label > 1)]
categorical_attr = c("sex", "cp", "fbs", "restecg", "exang", "thal")
for ( col in categorical_attr ) {
  ucihd[[col]] <- factor(ucihd[[col]], exclude=NULL)
}

# Train-test split.
is_test <- runif(nrow(ucihd)) > .7

# Train a random forest.
rf <- ranger(label ~ ., data=ucihd[!is_test], num.trees=300, probability=TRUE)

# Evaluate.
yhat <- predict(rf, ucihd[is_test])$predictions[,1]
roc(ucihd$label[is_test], yhat)

# Explain.
rf_predict_fn <- function(model, newdata) {
  p <- predict(model, newdata)
  p$predictions[,1]
}

predictor <- Predictor$new(
  model=rf, data=as.data.frame(ucihd[!is_test]), y="label",
  predict.fun=rf_predict_fn, type="prob")

# Global feature importance.
# TODO:
# All 0? Something wrong with the result...
imp = FeatureImp$new(predictor, loss="ce", compare="difference")
plot(imp)

# Accumulated local effects of a single feature.
age_ale = FeatureEffect$new(predictor, feature="age")
age_ale$plot()

# Shapley value.
sv <- Shapley$new(predictor, x.interest=as.data.frame(ucihd[is_test, -14][2]))
sv$results
sv$plot()

# LIME.
lime <- LocalModel$new(
  predictor, k=4,
  x.interest=ucihd[is_test, -14][2])
lime$results
plot(lime)
