#!/usr/bin/env python
'''Train a toy model using the iris dataset.'''

import lightgbm as lgb
from sklearn.datasets import load_iris


iris = load_iris()
data_train = lgb.Dataset(iris.data, iris.target, feature_name=iris.feature_names)

params = {
  'boosting_type': 'gbdt',
  'objective': 'multiclass',
  'num_class': 3,
  'metric': ['multi_logloss', 'multi_error'],
  'max_depth': 3,
  'num_leaves': 3,
  'learning_rate': .1,
}

bst = lgb.train(
  params,
  data_train,
  num_boost_round=10,
  valid_sets=[data_train],
)

bst.save_model('lambda_http_api/lgb.model')
