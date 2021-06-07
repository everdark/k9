try:
  import unzip_requirements
except ImportError:
  pass

import json

import numpy as np
import lightgbm as lgb


model = lgb.Booster(model_file='lgb.model')
label_names = np.array(['setosa', 'versicolor', 'virginica'])


def predict(event, context):

  body = json.loads(event['body'])
  x = [[body["sepal length"], body["sepal width"], body["petal length"], body["petal width"]]]
  yhat = model.predict(x)
  label = label_names[np.argmax(yhat, axis=1)]

  response = {
    'statusCode': 200,
    'body': json.dumps({
      'proba': yhat[0].tolist(),
      'label': label[0],
    }),
  }

  return response
