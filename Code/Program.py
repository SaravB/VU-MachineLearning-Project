import pandas as pd
import random
from datetime import datetime

from RandomForest import random_forest_algorithm, random_forest_predictions

df = pd.read_csv("mnist_train.csv", header=None)

# set label last
column_names = []
for column in df.columns:
    if column != 0:
        name = "pixel" + str(column)
        column_names.append(name)
    else:
        column_names.append("label")

df.columns = column_names

cols = df.columns.tolist()
cols = cols[1:] + cols[:1]
df = df[cols]


# split data in training and testing
def get_train_test_data(data_df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(data_df))

    indices = data_df.index.tolist()
    random.seed(0)
    test_indices = random.sample(population=indices, k=test_size)

    test_set = data_df.loc[test_indices]
    train_set = data_df.drop(test_indices)

    return train_set, test_set


train_df, validation_df = get_train_test_data(df, test_size=0.1)


def accuracy(predicted_labels, actual_labels):
    predictions_correct = predicted_labels == actual_labels

    return predictions_correct.mean()


n_trees = 100
n_bootstrap = len(train_df)
n_features = 30
dt_max_depth = 15

forest = random_forest_algorithm(train_df, n_trees=n_trees, n_bootstrap=n_bootstrap, n_features=n_features, dt_max_depth=dt_max_depth)

predictions = random_forest_predictions(test_df, forest)
prediction_accuracy = accuracy(predictions, test_df.label)

