import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, Conv1D, Attention, LayerNormalization, ReLU, Add
from ops import positional_encoding

def attention(q, k, v, mask = None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    #scaling res, recommended in the original paper, divide by sqrt(dim)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        #return x
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def __call__(self, v, k, q, mask = None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        #Here attetions head will be compute parallely
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        #we concatenate results from different attetion heads
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

def ffnn(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

def residual_block(layer_in, conv, features, reduce = False):
    strides_1 = 2 if reduce else 1
    # check if the number of filters needs to be increase, assumes channels last format
    merge_input = layer_in
    if layer_in.shape[-1] != features:
        merge_input = conv(features, 1, strides = strides_1, padding='same', kernel_initializer='he_normal')(layer_in)
        merge_input = BatchNormalization()(merge_input)
        merge_input = ReLU()(merge_input)
    # conv1
    conv1 = conv(features, 3, strides = strides_1, padding='same', kernel_initializer='he_normal')(layer_in)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    # conv2
    conv2 = conv(features, 3, padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    # add filters, assumes filters/channels last
    layer_out = Add()([conv2, merge_input])
    # activation function
    layer_out = ReLU()(layer_out)
    return layer_out
    
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model = 512, num_heads = 8, dff = 2048, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model             
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffnn = ffnn(d_model, dff)
        #normalization layers are trainable
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def __call__(self, x, training=None):
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        
        #residual
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffnn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        #residual
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffnn = ffnn(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        
    def __call__(self, x, enc_output, training, rank, look_ahead_mask = None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffnn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, embedding_layer,
                 maximum_position_encoding, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = embedding_layer
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
  
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def __call__(self, x, training = True):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        return x  # (batch_size, input_seq_len, d_model)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, embedding_layer,
                 maximum_position_encoding, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
    
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.embedding = embedding_layer
        
    def __call__(self, x, enc_output, look_ahead_mask = None, training = True):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#here we add info of patches position to patches!
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask)
      
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

