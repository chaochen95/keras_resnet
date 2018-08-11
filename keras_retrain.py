# _*_ coding: utf-8 _*_
import h5py
import tensorflow as tf
from keras.models import load_model
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.models import load_model
#base_model = ResNet50(weights = '/home/wcc/Desktop/py/weights.h5', include_top = True, pooling = 'avg')
model = load_model('/home/wcc/Desktop/py/weights8-10_00003.h5')  

# 训练的batch_size
batch_size = 16
# 训练的epoch
epochs = 1

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(
        rotation_range=40,#整数，数据提升时图片随机转动的角度
        width_shift_range=0.2,#浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        height_shift_range=0.2,#浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        #rescale=1./255,#重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        shear_range=0.2,#浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range=0.2,#浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        fill_mode='nearest',#‘constant’‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
        cval=0,#浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        horizontal_flip=True,#布尔值，进行随机水平翻转
        vertical_flip=False#布尔值，进行随机竖直翻转
        )

# 从文件中读取数据，目录结构应为train下面是各个类别的子目录，每个子目录中为对应类别的图像
train_generator = train_datagen.flow_from_directory('/home/wcc/Desktop/py/image_data/train_set', target_size = (224, 224), batch_size = batch_size)

# 训练图像的数量
image_numbers = train_generator.samples

# 输出类别信息
#for i , j in train_generator:
#        print(i,j)
''' 
print("-=---")'''
print (train_generator.class_indices)

# 生成测试数据
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory('/home/wcc/Desktop/py/image_data/val_set', target_size = (224, 224), batch_size = batch_size)
image_numbers_val = validation_generator.samples

# 训练模型
his = model.fit_generator(train_generator, 
                            steps_per_epoch = image_numbers // batch_size, 
                            epochs = epochs, 
                            validation_data = validation_generator, 
                            validation_steps = image_numbers_val // batch_size,
                            verbose=1)




 
model.save('weights8-10_00003.h5')
'''
for layer in model.layers:
    layer.trainable = False'''