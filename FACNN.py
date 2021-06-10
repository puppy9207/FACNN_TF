from tensorflow.python.keras.layers import Conv2D, Input,ReLU,UpSampling2D,Conv2DTranspose
from tensorflow.python.keras.models import Model
#d 56, s 12, m 4

def FACNN(scale,convert=False,width=960,heights=720):
    channels = 3
    PS = channels * (scale*scale)

    if convert == True:
        input_img = Input(shape=(width,heights,channels))
    else:
        input_img = Input(shape=(None,None, channels))

    model = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = Conv2D(128, (9, 9), padding='same', kernel_initializer='he_normal')(model)
    model = ReLU()(model)
    model = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = ReLU()(model)
    model = Conv2DTranspose(3, (9, 9), strides=(scale, scale), padding='same')(model)
    model_facnn = Model(input_img, model)
    return model_facnn