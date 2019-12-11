#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.util import to_pandas


# TODO: Find an interesting dataset for demo.

# All built-in datasets.
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
print(f"Available datasets: {list(dataset_recipes.keys())}")


url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0, index_col=0)

df.shape
df.head()

df.plot()

df.index.min()
df.index.max()

training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq="5min"
)
test_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
    freq = "5min"
)

estimator = DeepAREstimator(freq="5min", prediction_length=12, trainer=Trainer(epochs=10))
predictor = estimator.train(training_data=training_data)


for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which="both")


# TODO: Explore AWS DeepAR.
