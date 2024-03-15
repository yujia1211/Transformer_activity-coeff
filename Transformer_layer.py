import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Encoder_TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, en_dim, rate=0.1, name=None, **kwargs):
        super(Encoder_TransformerBlock, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.en_dim = en_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(en_dim, activation="tanh"), layers.Dense(en_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output, attn_weights = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'en_dim': self.en_dim,
            'rate': self.rate,
        })
        return config


class Decoder_TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, de_dim, rate=0.1, name=None, **kwargs):
        super(Decoder_TransformerBlock, self).__init__(name=name)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.de_dim = de_dim
        self.rate = rate

        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn1 = keras.Sequential(
            [layers.Dense(de_dim, activation="tanh"), layers.Dense(de_dim),]
        )
        self.ffn2 = keras.Sequential(
            [layers.Dense(de_dim, activation="tanh"), layers.Dense(de_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
        self.dropout4 = layers.Dropout(rate)

    def call(self, inputs, enc_output, look_ahead_mask=None, padding_mask=None, training=True):
        attn1_output, attn1_weights = self.att1(inputs, inputs, inputs, look_ahead_mask, return_attention_scores=True)
        attn1_output = self.dropout1(attn1_output, training=training)
        out1 = self.layernorm1(inputs + attn1_output)
        ffn1_output = self.ffn1(out1)
        ffn1_output = self.dropout2(ffn1_output, training=training)
        attn2_output, attn2_weights = self.att2(ffn1_output, enc_output, enc_output, padding_mask, return_attention_scores=True)
        attn2_output = self.dropout3(attn2_output, training=training)
        out2 = self.layernorm2(ffn1_output + attn2_output)
        ffn2_output = self.ffn2(out2)
        ffn2_output = self.dropout4(ffn2_output, training=training)
        out3 = self.layernorm3(out2 + ffn2_output)

        return out3, attn1_weights, attn2_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'de_dim': self.de_dim,
            'rate': self.rate,
        })
        return config

