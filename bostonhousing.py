import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import layers



X, y = load_boston(return_X_y=True)
X.shape
y.shape


scaler = MinMaxScaler()


X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)



dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset_train = dataset_train.batch(batch_size=32).shuffle(10000)
dataset_train = dataset_train.cache()
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

class NeuralNetwork(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = layers.Dense(units=26, activation='relu')
        self.layer2 = layers.Dense(units=15, activation='relu')
        self.layer3 = layers.Dense(units=1, activation='linear')
    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



model = NeuralNetwork()
loss_obj = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


for i in range(100):
    #loss_obj.reset_states()
    for X, y in dataset_train:
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = loss_obj(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
y_pred = model(X_test)
r2_score(y_test, y_pred.numpy())