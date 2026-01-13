from ultralytics import YOLOv10
# coding:utf-8
from ultralytics import YOLOv10

# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10s_UnblurNet2.yaml"
# 数据集配置文件
data_yaml_path = r'/root/yolov10-main/ultralytics/cfg/datasets/rebar.yaml'
# 预训练模型
pre_model_name = 'yolov10s.pt'

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLOv10("ultralytics/cfg/models/v10/yolov10s_UnblurNet2.yaml")
    #model = YOLOv10("ultralytics/cfg/models/v10/yolov10l.yaml").load('yolov10s.pt')_UnblurNet+small_head yolov10s_UnblurNet5.yaml
    # 训练模型
    results = model.train(data=data_yaml_path, epochs=200, batch=2, name='train_v10')


"""
import random
import numpy as np
import torch
from ultralytics import YOLOv10
# coding:utf-8

#
def set_seed(seed_value=42):
    #设置所有相关的随机数种子以确保可复现性
    # Python随机种子
    random.seed(seed_value)
    
    # Numpy随机种子
    np.random.seed(seed_value)
    
    # PyTorch随机种子
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多GPU
    
    # 设置CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10s_UnblurNet2.yaml"
# 数据集配置文件
data_yaml_path = r'/root/yolov10-main/ultralytics/cfg/datasets/rebar.yaml'

if __name__ == '__main__':
    # 设置随机数种子
    seed_value = 42  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    set_seed(seed_value)
    
    # 加载预训练模型
    model = YOLOv10(model_yaml_path)
    
    # 训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=200,
        batch=2,
        name='train_v10',
        seed=seed_value,  # YOLO内部的随机种子
        deterministic=True,  # 确保确定性
        # 其他参数...
    )
"""