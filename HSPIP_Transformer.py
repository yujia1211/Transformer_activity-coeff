import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Flatten, Dropout, LayerNormalization, \
    MultiHeadAttention, Concatenate, Conv2D, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import datetime
from sklearn.metrics import r2_score
from Transformer_layer import Encoder_TransformerBlock
from Transformer_layer import Decoder_TransformerBlock
import subfunc_1 as subs
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
#print(tf.keras.backend.floatx())
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#####model######
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

smiles = pd.read_csv('./DataPool/smiles_number.csv', header=None)
smiles = smiles.fillna(0)

sigma = pd.read_csv('./DataPool/sigma_true.csv', header=None).values

name = './hspip/'

if not os.path.exists(name):
    os.makedirs(name, exist_ok=True)

pos_index = []
for i in range(51):
    pos_index.append(i)
pos_index = np.array([pos_index] * 8175)

sigma = sigma.reshape(-1, 51, 1)
pos = to_categorical(pos_index, num_classes=51)


x_train, x_test = train_test_split(smiles, test_size=0.2, random_state=33)
y_train, y_test = train_test_split(sigma, test_size=0.2, random_state=33)
z_train, z_test = train_test_split(pos, test_size=0.2, random_state=33)

input = Input(batch_shape=(None, smiles.shape[1]))

emb = Embedding(input_dim=int(np.max(smiles.values)+1),
                input_length=49,
                output_dim=50,
                mask_zero=True)(input)

enc_output, _ = Encoder_TransformerBlock(num_heads=8, key_dim=50, en_dim=50, rate=0.1)(emb)
enc_output, _ = Encoder_TransformerBlock(num_heads=8, key_dim=50, en_dim=50, rate=0.1)(enc_output)
enc_output, _ = Encoder_TransformerBlock(num_heads=8, key_dim=50, en_dim=50, rate=0.2)(enc_output)

dec_input = Input(batch_shape=(None, 51, 51))

dec_output, _, _ = Decoder_TransformerBlock(num_heads=8, key_dim=51, de_dim=51, rate=0.1)(dec_input, enc_output)
dec_output, _, _ = Decoder_TransformerBlock(num_heads=8, key_dim=51, de_dim=51, rate=0.1)(dec_output, enc_output)
dec_output, attn1_weights, attn2_weights = Decoder_TransformerBlock(num_heads=8, key_dim=51, de_dim=51, rate=0.1)(dec_output, enc_output)

out = Dense(1, activation='relu')(dec_output)

model = Model(inputs=[input, dec_input], outputs=out)
model.summary()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

plot_model(model, to_file=name + 'plot_model.png', show_shapes=True, show_layer_names=True)

last_time = datetime.datetime.now()
#####model_train######
EarlyStopping_callback = EarlyStopping(patience=100, monitor='val_loss', mode='min')
CSVLogger_callback = CSVLogger(name + 'logs.csv')

weight_save_callback = ModelCheckpoint(name + 'sigma_Dense_model.h5', monitor='val_loss', verbose=0,
                                       save_best_only=True, mode='auto', period=1)

history = model.fit(x=[x_train, z_train],
                    y=y_train,
                    epochs=1000000,
                    batch_size=64,
                    validation_split=0.1,
                    verbose=1, shuffle=True,
                    callbacks=[EarlyStopping_callback, CSVLogger_callback, weight_save_callback])

current_time = datetime.datetime.now()

print("耗時： {}".format(current_time - last_time))

if not os.path.exists(name + 'sigma_Dense_model.h5'):
    model.save(name + 'sigma_Dense_model.h5')

#####圖輸出######
fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('total loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
fig_loss.savefig(name + 'loss.png')

model = load_model(name + 'sigma_Dense_model.h5', custom_objects={'Encoder_TransformerBlock': Encoder_TransformerBlock,
                                                                  'Decoder_TransformerBlock': Decoder_TransformerBlock,})
# model.compile(optimizer='Adam', loss='mse')

loss = model.evaluate([x_test, z_test], y_test)
pred = model.predict([x_test, z_test])
pred_csv = pred.reshape(-1, 51)
pred_list = pred.reshape(-1, 51)
y_test = y_test.reshape(-1, 51)

mse = []
for i in range(len(y_test)):
    mse_ = mean_squared_error(y_test[i, :], pred_list[i, :])
    mse.append(mse_)
mse = np.array(mse)


r2 = []
for i in range(len(y_test)):
    r2_ = subs.r2_draw(y_test[i, :], pred_list[i, :])
    r2.append(r2_)
r2 = np.array(r2)

r2_np = np.mean(r2)

print('\ntest loss:', loss)
print('\nR2_np:', r2_np)

test_per = np.percentile(r2, [95, 75, 50, 25, 5])
print('\n百分位數:', test_per)
