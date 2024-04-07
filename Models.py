from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten

class linear_encoder(Model):
    def __init__(self):
        super(linear_encoder, self).__init__()
        self.noise_dropout = Dropout(0.1)
        self.dense_1 = Dense(1024)
        self.leaky_relu_1 = LeakyReLU(alpha=0.2)
        self.dropout_1 = Dropout(0.2)
        self.dense_2 = Dense(1024)
        self.leaky_relu_2 = LeakyReLU(alpha=0.2)
        self.dropout_2 = Dropout(0.2)
        self.dense_3 = Dense(128)

    def call(self, inputs, **kwargs):
        x = self.noise_dropout(inputs)
        x = self.dense_1(x)
        x = self.leaky_relu_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dropout_1(x)
        outputs = self.dense_3(x)
        return outputs

class Data_decoder(Model):
    def __init__(self):
        super(Data_decoder, self).__init__()
        self.drop_rate = 0.2
        self.dense_1 = Dense(1024)
        self.leaky_relu_1 = LeakyReLU(alpha=0.2)
        self.dropout_1 = Dropout(self.drop_rate)
        self.dense_2 = Dense(1024)
        self.leaky_relu_2 = LeakyReLU(alpha=0.2)
        self.dropout_2 = Dropout(self.drop_rate)
        self.dense_3 = Dense(978, activation='tanh')

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.leaky_relu_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dropout_2(x)
        outputs = self.dense_3(x)

        return outputs

def con_model():
    input_A = keras.Input(shape=[978])
    input_B = keras.Input(shape=[978])
    encoder_A = linear_encoder()
    encoder_B = linear_encoder()

    l_A = encoder_A(input_A)
    l_B = encoder_B(input_B)

    input_de_A = keras.Input(shape=[128])
    input_de_B = keras.Input(shape=[128])
    decoder_A = Data_decoder()
    decoder_B = Data_decoder()

    out_A = decoder_A(input_de_A)
    out_B = decoder_B(input_de_B)

    model_A_en = keras.Model(input_A, l_A, name='en_A')
    model_A_de = keras.Model(input_de_A, out_A, name='de_A')
    model_B_en = keras.Model(input_B, l_B, name='en_B')
    model_B_de = keras.Model(input_de_B, out_B, name='de_B')
    #a2a,b2b,a2b,b2a
    return Model([input_A, input_B], [model_A_de(model_A_en(input_A)), model_B_de(model_B_en(input_B)),
                                      model_B_de(model_A_en(input_A)), model_A_de(model_B_en(input_B))])

def pre_model():
    input = keras.Input(shape=[978])
    l = linear_encoder()(input)
    input_l = keras.Input(shape=[128])
    output = Data_decoder()(input_l)

    model_en = keras.Model(input, l, name='en')
    model_de = keras.Model(input_l, output, name='de')

    return Model(input, model_de([model_en(input)]))

def multi(cell_nums):
    inputs = keras.Input(shape=[978, cell_nums])
    x = keras.layers.Dense(1, activation='tanh')(inputs)
    outputs = Flatten()(x)
    return Model(inputs, outputs)

def con_model__():
    input_A = keras.Input(shape=[978])
    input_B = keras.Input(shape=[978])
    encoder_A = linear_encoder()
    encoder_B = linear_encoder()

    l_A = encoder_A(input_A)
    l_B = encoder_B(input_B)

    input_de_A = keras.Input(shape=[128])
    input_de_B = keras.Input(shape=[128])
    decoder_A = Data_decoder()
    decoder_B = Data_decoder()

    out_A = decoder_A(input_de_A)
    out_B = decoder_B(input_de_B)

    model_A_en = keras.Model(input_A, l_A, name='en_A')
    model_A_de = keras.Model(input_de_A, out_A, name='de_A')
    model_B_en = keras.Model(input_B, l_B, name='en_B')
    model_B_de = keras.Model(input_de_B, out_B, name='de_B')
    #a2a,b2b,a2b,b2a
    return Model([input_A, input_B], [model_B_de(model_A_en(input_A)), model_A_de(model_B_en(input_B))])
