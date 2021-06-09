from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Conv2DTranspose, LeakyReLU,Input,ReLU
from tensorflow.python.keras.models import Model
#d 56, s 12, m 4

def FACNN(scale, d=56, s=12, m=4):
    channels = 3
    PS = channels * (scale*scale)

    input_img = Input(shape=(None,None, channels))

    model = Conv2D(d, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = ReLU()(model)

    model = Conv2D(s, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = ReLU()(model)
    for i in range(0,m):
        model = Conv2D(s, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = ReLU()(model)

    model = Conv2D(d, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = ReLU()(model)

    model = Conv2DTranspose(3, (9, 9), strides=(scale, scale), padding='same')(model)
    model_facnn = Model(input_img, model)
    return model_facnn