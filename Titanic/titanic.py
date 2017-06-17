from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.contrib import layers

IMAGE_SIZE = 96


def load_train_data():
    df = pd.read_csv("./train.csv")
    for c in df.columns:
        if df[c].dtype == object:
            print("convert ", df[c].name, " to string")
            df[c] = df[c].astype(basestring)
    print(df.dtypes)
    y = df["Survived"]
    cols = df.columns[2:]
    X = df[cols]
    print(X, y)
    return X, y

# def get_train_inputs():
#     X, y = load_train_data()
#     x = tf.constant(X)
#     y = tf.constant(y)
#     return x, y

X, y = load_train_data()


# def load_test_data():
#     df = pd.read_csv("./test.csv")
#     df_labels = pd.read_csv("./gender_submission.csv")
#     y = df_labels["Survived"].values
#     y = np.array(y)
#     X = df[df.columns[1:]].values
#     X = X.reshape(-1, 10, 1)
#     return X, y

classifier = learn.DNNClassifier(feature_columns=learn.infer_real_valued_columns_from_input(X),
                                 hidden_units=[10, 20, 10],
                                 n_classes=2,
                                 model_dir="iris_model")

classifier.fit(input_fn=load_train_data, steps=2000)

# X, y = load_test_data()
# accuracy_score = classifier.evaluate(input_fn=get_train_inputs, steps=1)["accuracy"]
# print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


#
#
# def new_samples():
#     return np.array(
#         [[6.4, 3.2, 4.5, 1.5],
#          [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
#
# predictions = list(classifier.predict(input_fn=new_samples))
#
# print("New Samples, Class Predictions: {}\n".format(predictions))
