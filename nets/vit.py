import math

import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import (Add, Conv2D, Dense, Dropout, Input, Lambda, Layer,
                          Reshape, Softmax)


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

#--------------------------------------------------------------------------------------------------------------------#
#   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
#
#   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
#   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
#   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
#--------------------------------------------------------------------------------------------------------------------#
class ClassToken(Layer):
    def __init__(self, cls_initializer='zeros', cls_regularizer=None, cls_constraint=None, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_initializer    = keras.initializers.get(cls_initializer)
        self.cls_regularizer    = keras.regularizers.get(cls_regularizer)
        self.cls_constraint     = keras.constraints.get(cls_constraint)

    def get_config(self):
        config = {
            'cls_initializer': keras.initializers.serialize(self.cls_initializer),
            'cls_regularizer': keras.regularizers.serialize(self.cls_regularizer),
            'cls_constraint': keras.constraints.serialize(self.cls_constraint),
        }
        base_config = super(ClassToken, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.cls = self.add_weight(
            shape       = (1, 1, self.num_features),
            initializer = self.cls_initializer,
            regularizer = self.cls_regularizer,
            constraint  = self.cls_constraint,
            name        = 'cls',
        )
        super(ClassToken, self).build(input_shape)

    def call(self, inputs):
        batch_size      = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(tf.broadcast_to(self.cls, [batch_size, 1, self.num_features]), dtype = inputs.dtype)
        return tf.concat([cls_broadcasted, inputs], 1)

#--------------------------------------------------------------------------------------------------------------------#
#   为网络提取到的特征添加上位置信息。
#   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
#   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
#--------------------------------------------------------------------------------------------------------------------#
class AddPositionEmbs(Layer):
    def __init__(self, image_shape, patch_size, pe_initializer='zeros', pe_regularizer=None, pe_constraint=None, **kwargs):
        super(AddPositionEmbs, self).__init__(**kwargs)
        self.image_shape        = image_shape
        self.patch_size         = patch_size
        self.pe_initializer     = keras.initializers.get(pe_initializer)
        self.pe_regularizer     = keras.regularizers.get(pe_regularizer)
        self.pe_constraint      = keras.constraints.get(pe_constraint)

    def get_config(self):
        config = {
            'pe_initializer': keras.initializers.serialize(self.pe_initializer),
            'pe_regularizer': keras.regularizers.serialize(self.pe_regularizer),
            'pe_constraint': keras.constraints.serialize(self.pe_constraint),
        }
        base_config = super(AddPositionEmbs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert (len(input_shape) == 3), f"Number of dimensions should be 3, got {len(input_shape)}"
        length  = (224 // self.patch_size) * (224 // self.patch_size) + 1
        self.pe = self.add_weight(
            # shape       = [1, input_shape[1], input_shape[2]],
            shape       = [1, length, input_shape[2]],
            initializer = self.pe_initializer,
            regularizer = self.pe_regularizer,
            constraint  = self.pe_constraint,
            name        = 'pos_embedding',
        )
        super(AddPositionEmbs, self).build(input_shape)

    def call(self, inputs):
        num_features = tf.shape(inputs)[2]

        cls_token_pe = self.pe[:, 0:1, :]
        img_token_pe = self.pe[:, 1: , :]

        img_token_pe = tf.reshape(img_token_pe, [1, (224 // self.patch_size), (224 // self.patch_size), num_features])
        img_token_pe = tf.image.resize_bicubic(img_token_pe, (self.image_shape[0] // self.patch_size, self.image_shape[1] // self.patch_size), align_corners=False)
        img_token_pe = tf.reshape(img_token_pe, [1, -1, num_features])
        
        pe = tf.concat([cls_token_pe, img_token_pe], axis = 1)

        return inputs + tf.cast(pe, dtype=inputs.dtype)

#--------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 点乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 点乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
#--------------------------------------------------------------------------------------------------------------------#
class Attention(Layer):
    def __init__(self, num_features, num_heads, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_features   = num_features
        self.num_heads      = num_heads
        self.projection_dim = num_features // num_heads

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] // 3)

    def call(self, inputs):
        #-----------------------------------------------#
        #   获得batch_size
        #-----------------------------------------------#
        bs      = tf.shape(inputs)[0]

        #-----------------------------------------------#
        #   b, 197, 3 * 768 -> b, 197, 3, 12, 64
        #-----------------------------------------------#
        inputs  = tf.reshape(inputs, [bs, -1, 3, self.num_heads, self.projection_dim])
        #-----------------------------------------------#
        #   b, 197, 3, 12, 64 -> 3, b, 12, 197, 64
        #-----------------------------------------------#
        inputs  = tf.transpose(inputs, [2, 0, 3, 1, 4])
        #-----------------------------------------------#
        #   将query, key, value划分开
        #   query     b, 12, 197, 64
        #   key       b, 12, 197, 64
        #   value     b, 12, 197, 64
        #-----------------------------------------------#
        query, key, value = inputs[0], inputs[1], inputs[2]
        #-----------------------------------------------#
        #   b, 12, 197, 64 @ b, 12, 197, 64 = b, 12, 197, 197
        #-----------------------------------------------#
        score           = tf.matmul(query, key, transpose_b=True)
        #-----------------------------------------------#
        #   进行数量级的缩放
        #-----------------------------------------------#
        scaled_score    = score / tf.math.sqrt(tf.cast(self.projection_dim, score.dtype))
        #-----------------------------------------------#
        #   b, 12, 197, 197 -> b, 12, 197, 197
        #-----------------------------------------------#
        weights         = tf.nn.softmax(scaled_score, axis=-1)
        #-----------------------------------------------#
        #   b, 12, 197, 197 @ b, 12, 197, 64 = b, 12, 197, 64
        #-----------------------------------------------#
        value          = tf.matmul(weights, value)

        #-----------------------------------------------#
        #   b, 12, 197, 64 -> b, 197, 12, 64
        #-----------------------------------------------#
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        #-----------------------------------------------#
        #   b, 197, 12, 64 -> b, 197, 768
        #-----------------------------------------------#
        output = tf.reshape(value, (tf.shape(value)[0], tf.shape(value)[1], -1))
        return output

def MultiHeadSelfAttention(inputs, num_features, num_heads, dropout, name):
    #-----------------------------------------------#
    #   qkv   b, 197, 768 -> b, 197, 3 * 768
    #-----------------------------------------------#
    qkv = Dense(int(num_features * 3), name = name + "qkv")(inputs)
    #-----------------------------------------------#
    #   b, 197, 3 * 768 -> b, 197, 768
    #-----------------------------------------------#
    x   = Attention(num_features, num_heads)(qkv)
    #-----------------------------------------------#
    #   197, 768 -> 197, 768
    #-----------------------------------------------#
    x   = Dense(num_features, name = name + "proj")(x)
    x   = Dropout(dropout)(x)
    return x

def MLP(y, num_features, mlp_dim, dropout, name):
    y = Dense(mlp_dim, name = name + "fc1")(y)
    y = Gelu()(y)
    y = Dropout(dropout)(y)
    y = Dense(num_features, name = name + "fc2")(y)
    return y

def TransformerBlock(inputs, num_features, num_heads, mlp_dim, dropout, name):
    #-----------------------------------------------#
    #   施加层标准化
    #-----------------------------------------------#
    x = LayerNormalization(epsilon=1e-6, name = name + "norm1")(inputs)
    #-----------------------------------------------#
    #   施加多头注意力机制
    #-----------------------------------------------#
    x = MultiHeadSelfAttention(x, num_features, num_heads, dropout, name = name + "attn.")
    x = Dropout(dropout)(x)
    #-----------------------------------------------#
    #   施加残差结构
    #-----------------------------------------------#
    x = Add()([x, inputs])

    #-----------------------------------------------#
    #   施加层标准化
    #-----------------------------------------------#
    y = LayerNormalization(epsilon=1e-6, name = name + "norm2")(x)
    #-----------------------------------------------#
    #   施加两次全连接
    #-----------------------------------------------#
    y = MLP(y, num_features, mlp_dim, dropout, name = name + "mlp.")
    y = Dropout(dropout)(y)
    #-----------------------------------------------#
    #   施加残差结构
    #-----------------------------------------------#
    y = Add()([x, y])
    return y

def VisionTransformer(input_shape = [224, 224], patch_size = 16, num_layers = 12, num_features = 768, num_heads = 12, mlp_dim = 3072, 
            classes = 1000, dropout = 0.1):
    #-----------------------------------------------#
    #   224, 224, 3
    #-----------------------------------------------#
    inputs      = Input(shape = (input_shape[0], input_shape[1], 3))
    
    #-----------------------------------------------#
    #   224, 224, 3 -> 14, 14, 768
    #-----------------------------------------------#
    x           = Conv2D(num_features, patch_size, strides = patch_size, padding = "valid", name = "patch_embed.proj")(inputs)
    #-----------------------------------------------#
    #   14, 14, 768 -> 196, 768
    #-----------------------------------------------#
    x           = Reshape(((input_shape[0] // patch_size) * (input_shape[1] // patch_size), num_features))(x)
    #-----------------------------------------------#
    #   196, 768 -> 197, 768
    #-----------------------------------------------#
    x           = ClassToken(name="cls_token")(x)
    #-----------------------------------------------#
    #   197, 768 -> 197, 768
    #-----------------------------------------------#
    x           = AddPositionEmbs(input_shape, patch_size, name="pos_embed")(x)
    #-----------------------------------------------#
    #   197, 768 -> 197, 768  12次
    #-----------------------------------------------#
    for n in range(num_layers):
        x = TransformerBlock(
            x,
            num_features= num_features,
            num_heads   = num_heads,
            mlp_dim     = mlp_dim,
            dropout     = dropout,
            name        = "blocks." + str(n) + ".",
        )
    x = LayerNormalization(
        epsilon=1e-6, name="norm"
    )(x)
    x = Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
    x = Dense(classes, name="head")(x)
    x = Softmax()(x)
    return keras.models.Model(inputs, x)
