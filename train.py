from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from FACNN import FACNN
import tensorflow as tf
from data_util import DataUtil
import os
import math

class Main():
    def __init__(self,epoch,train_path,crop_size,scale,batch):
        self.epoch = epoch
        self.train_path = train_path
        self.data_util = DataUtil(train_path,crop_size,scale)
        self.crop_size = crop_size
        self.scale = scale
        self.setDevice()
        self.batch = batch
        

    def setDevice(self):
        """ 환경 및 기기 설정 """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

    def log10(self,x):
        """ psnr에 들어가는 수식 """
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def calc_psnr(self,img1, img2):
        return 10. * self.log10(1. / tf.math.reduce_mean((img1 - img2) ** 2))
    
    def calc_ssim(self,img1,img2):
        return tf.image.ssim(img1,img2,max_val=1.0)

    def train(self):
        #{epoch}인 이유는 fit_generator에 들어갔을때 자동으로 입력되게 하기 위함
        filepath ="weights/model_{epoch}.h5"

        #모델을 저장하는 callback 함수
        checkpoint = ModelCheckpoint(filepath,monitor="loss",verbose=1,save_best_only=False,mode="min")
        callback_list = [checkpoint]

        #data Generator 불러오기
        training_generator = self.data_util

        #model load
        model = FACNN(2)

        #model loss와 optimizer, metrics에 저런식으로 함수를 넣으면 학습 중간에 지표를 보여준다.
        model.compile(loss="mean_absolute_error",optimizer="adam",metrics=[self.calc_psnr,self.calc_ssim])

        #model 학습
        model.fit_generator(generator=training_generator,use_multiprocessing=True,workers=8,callbacks=callback_list,epochs=self.epoch)

    def test(self):
        filepath = "weights/model_20.h5"
        model = FACNN(2)
        model.load_weights(filepath)
    
    def convertLite(self,filePath,output):
        filePath = filePath
        layer = FACNN(2,True,1920,1080)
        layer.load_weights(filePath)
        # new_model = tf.keras.models.load_model(filePath,custom_objects=
        # {
        #     "calc_psnr" : self.calc_psnr,
        #     "calc_ssim" : self.calc_ssim
        # })
        converter = tf.lite.TFLiteConverter.from_keras_model(layer)
        tflite_model = converter.convert()

        with open(output,'wb') as f:
            f.write(tflite_model)



if __name__ =="__main__":
    main = Main(30,"dataset/Flickr2K_HR",128,2,16)
    main.train()
    # main.convertLite("weights/model_1.h5","tfmodel.tflite")