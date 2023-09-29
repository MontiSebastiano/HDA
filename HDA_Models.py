from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, concatenate, GlobalMaxPooling2D, Flatten, Conv2D, multiply, LocallyConnected2D, Lambda, BatchNormalization
from keras.models import Model
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

def inception_model(image_size, batch_size):
    sex_input = Input(shape = (1), batch_size = batch_size, dtype = tf.float32, name = 'Sex')
    image_input = Input(shape = (image_size[0], image_size[1], 3), batch_size = batch_size, dtype = tf.float32, name = 'Images')

    x = Dense(32, activation = 'sigmoid')(sex_input) #sigmoid in place of relu

    inception_base = InceptionV3(weights='imagenet', include_top=False)
    for layer in inception_base.layers:
        layer.trainable = False

    #y = Rescaling(1./255)(image_input)
    y = inception_base(image_input)
    y = GlobalAveragePooling2D()(y)
    y = Dropout(0.5)(y) #NEW

    z = concatenate([y, x])
    z = Dense(1000, activation='tanh')(z) #sigmoid in place of tanh
    z = Dropout(0.25)(z) #NEW
    z = Dense(1000, activation='tanh')(z) #sigmoid in place of tanh
    #z = Dropout(0.25)(z) #NEW
    prediction = Dense(1, activation='linear', name = 'Prediction')(z)

    model = Model(inputs=[sex_input, image_input], outputs=prediction)

    return model


def xception_model(image_size, batch_size):
    sex_input = Input(shape = (1), batch_size = batch_size, dtype = tf.float32, name = 'Sex')
    image_input = Input(shape = (image_size[0], image_size[1], 3), batch_size = batch_size, dtype = tf.float32, name = 'Images')

    x = Dense(32, activation = 'sigmoid')(sex_input) #sigmoid in place of relu

    Xception_base = tf.keras.applications.xception.Xception(include_top = False, weights = 'imagenet')#input_shape = (500, 500, 3),
    for layer in Xception_base.layers:
        layer.trainable = False
    #model_1.trainable = True

    y = Xception_base(image_input)
    y = GlobalMaxPooling2D()(y)

    z = concatenate([y,x])
    z = Flatten()(z)
    z = Dense(10, activation = 'relu')(z)
    prediction = Dense(1, activation = 'linear')(z)

    model_xception = Model(inputs=[sex_input,image_input], outputs=prediction)

    return model_xception


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()

    def build(self, input_shape):
        # Create a trainable convolutional layer for computing attention weights
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        combined = tf.concat([avg_pool, max_pool], axis=3)
        attention = self.conv(combined)
        return attention * inputs
    

class MultiHeadSpatialAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, kernel_size=3):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv_heads = []
        for _ in range(self.num_heads):
            self.conv_heads.append(tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation='sigmoid', padding='same', use_bias=False))

    def call(self, inputs):
        head_outputs = []
        for i in range(self.num_heads):
            attention_weights = self.conv_heads[i](inputs)
            head_output = attention_weights * inputs
            head_outputs.append(head_output)
        return tf.concat(head_outputs, axis=-1)
    

def attention_model(image_size, batch_size):
    sex_input = Input(shape = (1), batch_size = batch_size, dtype = tf.float32, name = 'Sex')
    image_input = Input(shape = (image_size[0], image_size[1], 3), batch_size = batch_size, dtype = tf.float32, name = 'Images')

    x = Dense(32, activation = 'sigmoid')(sex_input) #sigmoid in place of relu


    Xception_base = tf.keras.applications.xception.Xception(include_top = False, weights = 'imagenet')#input_shape = (500, 500, 3),
    for layer in Xception_base.layers:
        layer.trainable = False
    #model_1.trainable = True

    #attention_layer = SpatialAttention()
    attention_layer = MultiHeadSpatialAttention(num_heads=8, kernel_size=5)

    y = Xception_base(image_input)
    y = attention_layer(y)
    y = GlobalMaxPooling2D()(y)

    z = concatenate([y,x])
    z = Flatten()(z)
    z = Dense(10, activation = 'relu')(z)
    prediction = Dense(1, activation = 'linear')(z)

    model_attention = Model(inputs=[sex_input,image_input], outputs=prediction)

    return model_attention

class VGGAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, in_features_l, in_features_g, attn_features):
        super(VGGAttentionBlock, self).__init__()
        self.W_l = Conv2D(filters=attn_features, kernel_size=(1, 1), padding='valid', use_bias=False)
        self.W_g = Conv2D(filters=attn_features, kernel_size=(1, 1), padding='valid', use_bias=False)
        self.phi = Conv2D(filters=1, kernel_size=(1, 1), padding='valid', activation='relu', use_bias=True)

    def call(self, l, g):
        N, H, W, C = l.shape
        l_ = self.W_l(l)
        g_ = self.W_g(g)

        g_ = tf.image.resize(g_, size=(W, H), method='bilinear')

        c = self.phi(l_ + g_)  # batch_sizexWxHx1
        
        a = tf.nn.softmax(tf.reshape(c, (N, -1, 1)), axis=1)
        a = tf.reshape(a, (N, H, W, 1))

        f = tf.multiply(a, l)
        f = tf.reshape(f, (N, C, -1))

        output = tf.reduce_sum(tf.reshape(f, (N, C, -1)), axis=2)

        return output
    
def VGGAttention_model(image_size, batch_size):
    sex_input = Input(shape = (1), batch_size = batch_size, dtype = tf.float32, name = 'Sex')
    image_input = Input(shape = (image_size[0], image_size[1], 3), batch_size = batch_size, dtype = tf.float32, name = 'Images')

    x = Dense(32, activation = 'sigmoid')(sex_input)

    vgg16_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

    for layer in vgg16_base.layers:
        layer.trainable = False

    vgg_pool3 = tf.keras.models.Model(inputs=vgg16_base.input, outputs=vgg16_base.get_layer('block3_pool').output)
    intermediate_features3 = vgg_pool3(image_input)

    vgg_pool4 = tf.keras.models.Model(inputs=vgg16_base.input, outputs=vgg16_base.get_layer('block4_pool').output)
    intermediate_features4 = vgg_pool4(image_input)

    vgg_output = vgg16_base(image_input)
    f = GlobalMaxPooling2D(data_format='channels_last')(vgg_output)

    attn1 = VGGAttentionBlock(256, 512, 256)
    attn2 = VGGAttentionBlock(512, 512, 256)

    f1 = attn1(intermediate_features3, vgg_output)
    f2 = attn2(intermediate_features4, vgg_output)

    z = concatenate([f, f1, f2, x])

    prediction = Dense(1, activation='linear', name = 'Prediction')(z)

    model_VGGAttention = Model(inputs=[sex_input,image_input], outputs=prediction)

    return model_VGGAttention


def resnet_model(patch_size, batch_size):
    sex_input = Input(shape=(1), batch_size=batch_size, dtype=tf.float32, name='Sex')
    image_input = Input(shape=(patch_size[0], patch_size[1], 3), batch_size=batch_size, dtype=tf.float32, name='Images')

    #x = Dense(32, activation='sigmoid')(sex_input)  # sigmoid in place of relu

    ResNet50_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')#, input_shape=(224, 224, 3))

    for layer in ResNet50_base.layers:
        layer.trainable = False

    y = ResNet50_base(image_input)
    y = GlobalAveragePooling2D()(y)

    z = concatenate([y, sex_input])
    z = Flatten()(z)
    z = Dense(14, activation='relu')(z)   #cambiato da 10 a 14
    prediction = Dense(1, activation='linear')(z)

    model_resnet50 = Model(inputs=[sex_input, image_input], outputs=prediction)

    return model_resnet50