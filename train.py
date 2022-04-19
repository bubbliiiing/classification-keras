import datetime
import os

import numpy as np
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils.multi_gpu_utils import multi_gpu_model

from nets import freeze_layers, get_model_from_name
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ParallelModelCheckpoint)
from utils.dataloader import ClsDatasets
from utils.utils import get_classes, get_lr_scheduler

if __name__ == "__main__":
    #---------------------------------------------------------------------#
    #   train_gpu   训练用到的GPU
    #               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/cls_classes.txt' 
    #------------------------------------------------------#
    #   input_shape     输入的shape大小
    #------------------------------------------------------#
    input_shape     = [224, 224]
    #------------------------------------------------------#
    #   所用模型种类：
    #   mobilenet、resnet50、vgg16、vit
    #------------------------------------------------------#
    backbone        = "mobilenet"
    #------------------------------------------------------#
    #   当使用mobilenet的alpha值
    #   仅在backbone='mobilenet'的时候有效
    #------------------------------------------------------#
    alpha           = 0.25
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/mobilenet_2_5_224_tf_no_top.h5"
        
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从0开始训练：
    #       Adam：
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    #       其中：UnFreeze_Epoch尽量不小于300。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 32
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers         = 1
    
    #------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   test_annotation_path    验证图片路径和标签（使用测试集代替验证集）
    #------------------------------------------------------#
    train_annotation_path   = "cls_train.txt"
    test_annotation_path    = 'cls_test.txt'

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    #------------------------------------------------------#
    #   获取classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   创建分类模型
    #------------------------------------------------------#
    if backbone == "mobilenet":
        model_body = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes, alpha=alpha)
    else:
        model_body = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes)
    
    if model_path != "":
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if ngpus_per_node > 1:
        model   = multi_gpu_model(model_body, gpus=ngpus_per_node)
    else:
        model   = model_body

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        if Freeze_Train:
            freeze_layers = freeze_layers[backbone]
            for i in range(freeze_layers): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
            
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
        
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == 'vit':
            nbs             = 128
            lr_limit_max    = 3e-4 if optimizer_type == 'adam' else 5e-2
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['categorical_accuracy'])

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        
        train_dataloader    = ClsDatasets(train_lines, input_shape, batch_size, num_classes, train = True)
        val_dataloader      = ClsDatasets(val_lines, input_shape, batch_size, num_classes, train = False)

        #-------------------------------------------------------------------------------#
        #   训练参数的设置
        #   logging         用于设置tensorboard的保存地址
        #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        #   lr_scheduler       用于设置学习率下降的方式
        #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
        #-------------------------------------------------------------------------------#
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        if ngpus_per_node > 1:
            checkpoint      = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
        else:
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks       = [logging, loss_history, checkpoint, lr_scheduler]

        if start_epoch < end_epoch:
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
        #---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        #---------------------------------------#
        if Freeze_Train:
            batch_size  = Unfreeze_batch_size
            start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
            end_epoch   = UnFreeze_Epoch
                
            #-------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            #-------------------------------------------------------------------#
            nbs             = 64
            lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
            if backbone == 'vit':
                nbs             = 128
                lr_limit_max    = 3e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            #---------------------------------------#
            #   获得学习率下降的公式
            #---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler]
            
            for i in range(len(model_body.layers)): 
                model_body.layers[i].trainable = True
            model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['categorical_accuracy'])

            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataloader.batch_size    = Unfreeze_batch_size
            val_dataloader.batch_size      = Unfreeze_batch_size

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
