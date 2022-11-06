import math

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Add, Conv2D, Dense, Dropout, GlobalAveragePooling1D,
                          Input, Lambda, Layer, Reshape, Softmax)


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob       = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor   = keep_prob
    shape           = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
    random_tensor   += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor   = tf.floor(random_tensor)
    output          = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output

class DropPath(keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)

#--------------------------------------#
#   LayerNormalization
#   层标准化的实现
#--------------------------------------#
class LayerNormalization(keras.layers.Layer):
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

#--------------------------------------#
#   Gelu激活函数的实现
#   利用近似的数学公式
#--------------------------------------#
class Gelu(Layer):
    def __init__(self, **kwargs):
        super(Gelu, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return 0.5 * inputs * (1 + tf.tanh(tf.sqrt(2 / math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))))

    def get_config(self):
        config = super(Gelu, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class Mlp():
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., name=""):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1        = Dense(hidden_features, name=name + '.fc1')
        self.act_layer  = Gelu()
        self.fc2        = Dense(out_features, name=name + '.fc2')
        self.drop       = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x

class SwinTransformerBlock_pre(keras.layers.Layer):
    def __init__(self, input_resolution, window_size=7, shift_size=0):
        super().__init__()
        self.input_resolution   = input_resolution
        self.window_size        = window_size
        self.shift_size         = shift_size
    
    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(tf.not_equal(attn_mask, 0), -100.0 * tf.ones_like(attn_mask), attn_mask)
            attn_mask = tf.where(tf.equal(attn_mask, 0), tf.zeros_like(attn_mask), attn_mask)
            
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False)
        else:
            self.attn_mask = None

        self.built = True

    def compute_output_shape(self, input_shape):
        return (None, self.window_size * self.window_size, input_shape[2])

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        
        x = tf.reshape(x, shape=[-1, H, W, C])

        # 56, 56, 96
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # 56, 56, 96 -> 8, 7, 8, 7, 96 -> 8, 8, 7, 7, 96 -> 64, 7, 7, 96 -> 64, 49, 96
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=[-1, self.window_size * self.window_size, C])
        return x_windows


class WindowAttention_pre(keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0, name=""):
        super().__init__(name=name)
        self.dim            = dim
        self.window_size    = window_size
        self.num_heads      = num_heads

        head_dim            = dim // num_heads
        self.scale          = qk_scale or head_dim ** -0.5
        self.attn_drop      = Dropout(attn_drop)
    
    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            f'attn/relative_position_bias_table',
            shape       = ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer = tf.initializers.zeros(), 
            trainable   = True
        )

        coords_h        = np.arange(self.window_size[0])
        coords_w        = np.arange(self.window_size[1])
        coords          = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten  = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)

        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'attn/relative_position_index')
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] // 3)

    def call(self, x, mask=None):
        B_, N, C    = x.get_shape().as_list()
        C           = C // 3
        # [B_, N, C] -> [B_, N, 3 * C] -> [B_, N, 3, num_heads, C / num_heads] -> [3, B_, num_heads, N, C / num_heads]
        qkv         = tf.transpose(tf.reshape(x, shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        # [B_, num_heads, N, C / num_heads]
        q, k, v     = qkv[0], qkv[1], qkv[2]

        # [B_, num_heads, N, N]
        q       = q * self.scale
        attn    = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))

        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        # [B_, num_heads, N, N]
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + \
                tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            # [B_, num_heads, N, N]
            attn = tf.nn.softmax(attn, axis=-1)
        
        attn = self.attn_drop(attn)

        # [B_, num_heads, N, C / num_heads] -> [B_, N, num_heads, C / num_heads] -> [B_, N, C]
        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        return x

class WindowAttention():
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., name=""):
        super().__init__()

        self.qkv            = Dense(dim * 3, use_bias=qkv_bias, name=name + ".qkv")
        self.pre            = WindowAttention_pre(dim, window_size, num_heads, qk_scale, attn_drop, name=name)

        self.proj           = Dense(dim, name=name + ".proj")
        self.proj_drop      = Dropout(proj_drop)


    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        x = self.qkv(x)
        x = self.pre(x, mask = mask)

        # [B_, N, C] -> [B_, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock_post(keras.layers.Layer):
    def __init__(self, dim, input_resolution, window_size=7, shift_size=0):
        super().__init__()
        self.dim                = dim
        self.input_resolution   = input_resolution
        self.window_size        = window_size
        self.shift_size         = shift_size

    def compute_output_shape(self, input_shape):
        return (None, self.input_resolution[0] * self.input_resolution[1], input_shape[2])

    def call(self, x):
        H, W = self.input_resolution

        # 64, 49, 97 -> 64, 7, 7, 97 -> 8, 8, 7, 7, 96 -> 8, 7, 8, 7, 96 -> 56, 56, 96
        attn_windows = tf.reshape(x, shape=[-1, self.window_size, self.window_size, self.dim])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, self.dim)
        # 56, 56, 96
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        
        # 56 * 56, 96
        x = tf.reshape(x, shape=[-1, H * W, self.dim])
        return x

class SwinTransformerBlock():
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., name=""):
        super().__init__()
        self.dim                = dim
        self.input_resolution   = input_resolution
        self.num_heads          = num_heads
        self.window_size        = window_size
        self.shift_size         = shift_size
        self.mlp_ratio          = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.dim                = dim
        self.input_resolution   = input_resolution
        self.num_heads          = num_heads
        self.window_size        = window_size
        self.shift_size         = shift_size
        self.mlp_ratio          = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1              = LayerNormalization(epsilon=1e-5, name=name + ".norm1")
        self.pre                = SwinTransformerBlock_pre(self.input_resolution, self.window_size, self.shift_size)
        self.attn               = WindowAttention(
            dim, 
            window_size = (self.window_size, self.window_size), 
            num_heads   = num_heads,
            qkv_bias    = qkv_bias, 
            qk_scale    = qk_scale, 
            attn_drop   = attn_drop, 
            proj_drop   = drop, 
            name        = name + ".attn"
        )
        self.post               = SwinTransformerBlock_post(self.dim, self.input_resolution, self.window_size, self.shift_size)
        self.drop_path          = DropPath(drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2              = LayerNormalization(epsilon=1e-5, name=name + ".norm2")
        mlp_hidden_dim          = int(dim * mlp_ratio)
        self.mlp                = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, name=name + ".mlp")
        self.add                = Add()


    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        # 56, 56, 96

        shortcut = x

        x = self.norm1(x)
        x = self.pre(x)
        # 64, 49, 97 -> 64, 49, 97
        x = self.attn.call(x, mask=self.pre.attn_mask)
        x = self.post(x)

        # FFN
        # 56 * 56, 96
        x = self.add([shortcut, self.drop_path(x)])
        x = self.add([x, self.drop_path(self.mlp.call(self.norm2(x)))])
        return x

class PatchMerging(keras.layers.Layer):
    def __init__(self, input_resolution):
        super().__init__()
        self.input_resolution   = input_resolution

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 4, input_shape[2] * 4)
        
    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # 56, 56, 96
        x = tf.reshape(x, shape=[-1, H, W, C])

        # 28, 28, 96
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # 28, 28, 384
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        # 784, 384
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        return x

def BasicLayer(
    x, dim, input_resolution, depth, num_heads, window_size,
    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., name=""
):
    for i in range(depth):
        x = SwinTransformerBlock(
                    dim                 = dim, 
                    input_resolution    = input_resolution,
                    num_heads           = num_heads, 
                    window_size         = window_size,
                    shift_size          = 0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio           = mlp_ratio,
                    qkv_bias            = qkv_bias, 
                    qk_scale            = qk_scale,
                    drop                = drop, 
                    attn_drop           = attn_drop,
                    drop_path_prob      = drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                    name                = name + ".blocks." + str(i),
                ).call(x)
    return x

def build_model(input_shape = [224, 224], patch_size=(4, 4), classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
    #-----------------------------------------------#
    #   224, 224, 3
    #-----------------------------------------------#
    inputs = Input(shape = (input_shape[0], input_shape[1], 3))
    
    #-----------------------------------------------#
    #   224, 224, 3 -> 56, 56, 768
    #-----------------------------------------------#
    x = Conv2D(embed_dim, patch_size, strides = patch_size, padding = "valid", name = "patch_embed.proj")(inputs)
    #-----------------------------------------------#
    #   56, 56, 768 -> 3136, 768
    #-----------------------------------------------#
    x = Reshape(((input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[0]), embed_dim))(x)
    x = LayerNormalization(epsilon=1e-5, name = "patch_embed.norm")(x)
    x = Dropout(drop_rate)(x)

    num_layers          = len(depths)
    patches_resolution  = [input_shape[0] // patch_size[0], input_shape[1] // patch_size[1]]
    dpr                 = [x for x in np.linspace(0., drop_path_rate, sum(depths))]
    #-----------------------------------------------#
    #   3136, 768 -> 3136, 49 
    #-----------------------------------------------#
    for i_layer in range(num_layers):
        dim                 = int(embed_dim * 2 ** i_layer)
        input_resolution    = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
        x = BasicLayer(
            x,
            dim                 = dim,
            input_resolution    = input_resolution,
            depth               = depths[i_layer],
            num_heads           = num_heads[i_layer],
            window_size         = window_size,
            mlp_ratio           = mlp_ratio,
            qkv_bias            = qkv_bias, qk_scale=qk_scale,
            drop                = drop_rate, attn_drop=attn_drop_rate,
            drop_path_prob      = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            name                = "layers." + str(i_layer)
        )
        if (i_layer < num_layers - 1):
            x   = PatchMerging(input_resolution)(x)
            x   = LayerNormalization(epsilon=1e-5, name = "layers." + str(i_layer) + ".downsample.norm")(x)
            x   = Dense(2 * dim, use_bias=False, name = "layers." + str(i_layer) + ".downsample.reduction")(x)

    x = LayerNormalization(epsilon=1e-5, name="norm")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(classes, name="head")(x)
    x = Softmax()(x)
    return keras.models.Model(inputs, x)

def swin_transformer_tiny(input_shape=[224, 224], classes=1000):
    model = build_model(input_shape, classes=classes, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], embed_dim=96, drop_path_rate=0.2)
    return model

def swin_transformer_small(input_shape=[224, 224], classes=1000):
    model = build_model(input_shape, classes=classes, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], embed_dim=96, drop_path_rate=0.3)
    return model

def swin_transformer_base(input_shape=[224, 224], classes=1000):
    model = build_model(input_shape, classes=classes, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], embed_dim=128, drop_path_rate=0.5)
    return model