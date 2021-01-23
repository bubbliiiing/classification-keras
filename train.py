import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image

from nets.mobilenet import MobileNet
from nets.resnet50 import ResNet50
from nets.vgg16 import VGG16
from utils.utils import get_random_data, letterbox_image

get_model_from_name = {
    "mobilenet" : MobileNet,
    "resnet50"  : ResNet50,
    "vgg16"     : VGG16,
}

freeze_layers = {
    "mobilenet" : 81,
    "resnet50"  : 173,
    "vgg16"     : 19,
}

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def _preprocess_input(x,):
    x /= 127.5
    x -= 1.
    return x

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#----------------------------------------#
#   训练数据生成器
#----------------------------------------#
def generate_arrays_from_file(lines, batch_size, input_shape, num_classes, random=True):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            annotation_path = lines[i].split(';')[1].split()[0]
            img = Image.open(annotation_path)
            
            if random:
                img = get_random_data(img, [input_shape[0],input_shape[1]])
            else:
                img = letterbox_image(img, [input_shape[0],input_shape[1]])

            img = np.array(img).astype(np.float32)
            img = _preprocess_input(img)

            X_train.append(img)
            Y_train.append(int(lines[i].split(';')[0]))
            i = (i+1) % n

        X_train = np.array(X_train)
        X_train = X_train.reshape([-1,input_shape[0],input_shape[1],input_shape[2]])
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes=num_classes)   
        yield (X_train, Y_train)

#----------------------------------------#
#   主函数
#----------------------------------------#
if __name__ == "__main__":
    log_dir = "./logs/"
    #------------------------------#
    #   输入的图片大小
    #------------------------------#
    input_shape = [224,224,3]
    #------------------------------#
    #   所用模型种类
    #------------------------------#
    backbone = "mobilenet"
    #------------------------------#
    #   当使用mobilenet的alpha值
    #------------------------------#
    alpha = 0.25

    #----------------------------------------------------#
    #   训练自己的数据集的时候一定要注意修改classes_path
    #   修改成自己对应的种类的txt
    #----------------------------------------------------#
    classes_path = './model_data/cls_classes.txt' 
    class_names = get_classes(classes_path)
    num_classes = len(class_names)  

    assert backbone in ["mobilenet", "resnet50", "vgg16"]

    if backbone == "mobilenet":
        model = get_model_from_name[backbone](input_shape=input_shape, classes=num_classes, alpha=alpha)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, classes=num_classes)

    #----------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #----------------------------------------------------#
    with open(r"./cls_train.txt","r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   不使用预训练权重效果会很差
    #------------------------------------------------------#
    model_path = "model_data/mobilenet_2_5_224_tf_no_top.h5"
    model.load_weights(model_path,by_name=True,skip_mismatch=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   tensorboard表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    tensorboard = TensorBoard(log_dir=log_dir)
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    for i in range(freeze_layers[backbone]):
        model.layers[i].trainable = False

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        BATCH_SIZE = 16
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50

        model.compile(loss = 'categorical_crossentropy',
                optimizer = Adam(lr = Lr),
                metrics = ['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        
        model.fit_generator(generate_arrays_from_file(lines[:num_train], BATCH_SIZE, input_shape=input_shape, num_classes=num_classes, random=True),
                steps_per_epoch=max(1, num_train//BATCH_SIZE),
                validation_data=generate_arrays_from_file(lines[num_train:], BATCH_SIZE, input_shape=input_shape, num_classes=num_classes, random=False),
                validation_steps=max(1, num_val//BATCH_SIZE),
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])


    for i in range(freeze_layers[backbone]):
        model.layers[i].trainable = True

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        BATCH_SIZE = 8
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 100

        model.compile(loss = 'categorical_crossentropy',
                optimizer = Adam(lr = Lr),
                metrics = ['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        
        model.fit_generator(generate_arrays_from_file(lines[:num_train], BATCH_SIZE, input_shape=input_shape, num_classes=num_classes, random=True),
                steps_per_epoch=max(1, num_train//BATCH_SIZE),
                validation_data=generate_arrays_from_file(lines[num_train:], BATCH_SIZE, input_shape=input_shape, num_classes=num_classes, random=False),
                validation_steps=max(1, num_val//BATCH_SIZE),
                epochs=Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])



