#------------------------------------------------------#
#   该eval文件会自动计算
#   Top1 acc
#   Top5 acc
#   Recall
#   Precision
#   结果会保留在metrics_out文件夹中
#------------------------------------------------------#
import os

import numpy as np

from classification import Classification, cvtColor, preprocess_input
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5

#------------------------------------------------------#
#   test_annotation_path    测试图片路径和标签
#------------------------------------------------------#
test_annotation_path    = 'cls_test.txt'
#------------------------------------------------------#
#   metrics_out_path        指标保存的文件夹
#------------------------------------------------------#
metrics_out_path        = "metrics_out"

class Eval_Classification(Classification):
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        
        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        preds = self.model.predict(image_data)[0]
        return preds
    
if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification()
    
    with open("./cls_test.txt","r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
