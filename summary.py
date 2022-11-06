#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets import get_model_from_name
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape     = [224, 224]
    num_classes     = 1000
    backbone        = "mobilenetv1"
    
    model = get_model_from_name[backbone]([input_shape[0], input_shape[1], 3], classes=num_classes)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)

    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    for i,layer in enumerate(model.layers):
        print(i,layer.name)
