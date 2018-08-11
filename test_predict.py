# _*_ coding: utf-8 _*_
import h5py,os,sys,shutil
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
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import heapq
from keras.models import load_model
#base_model = ResNet50(weights = '/home/wcc/Desktop/py/weights.h5', include_top = True, pooling = 'avg')
model = load_model('/home/wcc/Desktop/py/weights8-11_trainable_rescale.h5')  

dic = {'aircraft': 0, 
        'alien': 1, 
        'animal': 2, 
        'balloon': 3, 
        'black people': 4, 
        'boxing': 5, 
        'carcrash': 6, 
        'church': 7, 
        'desert': 8, 
        'doctor': 9, 
        'dolphin': 10, 
        'earthquake': 11, 
        'farm': 12, 
        'farmer': 13, 
        'gangster': 14, 
        'grassland': 15, 
        'hospital': 16, 
        'mahjong': 17, 
        'monsters': 18, 
        'music band': 19, 
        'ocean': 20, 
        'penguin': 21, 
        'piano': 22, 
        'police': 23, 
        'princess': 24, 
        'racing': 25, 
        'robot': 26, 
        'running': 27, 
        'soldier': 28, 
        'tall building': 29, 
        'tank': 30, 
        'tiger': 31, 
        'train': 32, 
        'tsunami': 33, 
        'unbralla': 34, 
        'universe': 35, 
        'war': 36, 
        'west': 37, 
        'wife': 38}
new_dict = {v : k for k, v in dic.items()}

err = 0
err_top5 = 0
img_num = 0
error_path = '/home/wcc/Desktop/py/image_data/error_test'
path = '/home/wcc/Desktop/py/image_data/val_set'
#path = '/home/wcc/Desktop/py/image_data/train_set'
#error_path = '/home/wcc/Desktop/py/image_data/error_train'
if not os.path.exists(error_path):
    os.makedirs(error_path)
class_path = os.listdir(path)
for j in class_path:
    image_path = path +'/'+j
    img_list = os.listdir(image_path)
    for i in img_list:
        top5 = []
        top5_tem = []
        file_path = image_path +'/'+i
        # 加载图像
        img = image.load_img(file_path, target_size=(224, 224))

        # 图像预处理
        x = image.img_to_array(img)
        
        x = preprocess_input(x,mode="caffe")
        x = image.array_to_img(x)
        
        b, g, r = x.split()
        x = Image.merge("RGB", (r, g, b))
        x = image.img_to_array(x)
        x /= 255
        x = np.expand_dims(x, axis=0)
        #print(x)
        # 对图像进行分类
        preds = model.predict(x)
        #print(decode_predictions(preds, top=3)[0])
        #print (j +'/'+ i)
        #print(preds.tolist()[0])
        tem = preds.tolist()[0]
        

        preds_sum = 0
        for k in tem:
            preds_sum += k
        preds_prob = heapq.nlargest(5, tem)
        preds_prob_top5 =[]
        #print (j +'/'+ i )
        #print(preds_sum)
        for k in preds_prob:
            preds_prob_top5.append(round(k/preds_sum,2))
        
        #print(preds_prob_top5)
        max_num_index_list = map(tem.index, heapq.nlargest(5, tem))
        
        preds = int(np.argmax(preds,1))
        
        
        tem = list(max_num_index_list)
        #print(tem)
        for k in tem:
            top5_tem.append(new_dict[int(k)])
        #print(str(top5))
        #print(type(j))
        #print(j)
        #print(type(i))
        
        preds_class = new_dict[preds]
        
        zip_top5 = zip(top5_tem,preds_prob_top5)
        for k in zip_top5:
            #print(type(i))
            top5.extend(list(k))
        #print(type(top5))
        #break
        # 输出预测概率
        img_num += 1
        
        #break
        if dic[j] != preds:
            if not os.path.exists(error_path+'/'+preds_class):
                os.makedirs(error_path+'/'+preds_class)
            shutil.copy(path+'/'+ j + '/' + i, error_path+'/'+preds_class+'/'+ j +'_'+'top5:'+str(top5)+ i)
            print (j +'/'+ i + '-----'+ 'predicted:' + preds_class)
            err += 1
            tem = tem[1:]
            if dic[j] not in tem:
                err_top5 += 1
    #break
print('top1 acc:'+str((img_num - err)/img_num))
print('top5 acc:'+str((img_num - err_top5)/img_num))
