import pathlib
import torchvision.transforms as transforms
from degradation import degradation_pipeline
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import os
def degradation(image):
    image = Image.fromarray(degradation_pipeline(np.array(image)).astype(np.uint8))
    return image

def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

#pytorch dataloader와 매우 유사.
class DataUtil(Sequence):
    def __init__(self,train_path,crop_size,scale,batch=16):
        self.batch = batch
        self.filenames = [os.path.join(train_path, x) for x in os.listdir(os.path.join(train_path)) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.Resize((crop_size // scale, crop_size // scale), interpolation=Image.BICUBIC),
            transforms.Lambda(degradation),
        ])
        self.hr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        
        self.x_dim = (crop_size//scale,crop_size//scale,3)
        self.y_dim = (crop_size,crop_size,3)
        self.on_epoch_end()

    def __getitem__(self, index):
        # Batch를 위한 빈 numpy 배열
        X = np.empty((self.batch,*self.x_dim))
        y = np.empty((self.batch,*self.y_dim))

        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        file_lists = [self.filenames[k] for k in indexes]

        for i,name in enumerate(file_lists):
            im = cv2.imread(name, 3)
            hr = self.hr_transforms(im)
            lr = self.lr_transforms(hr)
            hr = np.asarray(hr)/255.
            lr = np.asarray(lr)/255.
            X[i] = lr
            y[i] = hr

        
        return X , y

    #총 길이 
    def __len__(self):
        return len(self.filenames)//int(self.batch)

    #에포크가 끝나면 불러옴
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filenames))