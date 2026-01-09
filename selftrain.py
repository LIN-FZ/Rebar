import random
import numpy as np
import torch
from ultralytics import YOLOv10
# coding:utf-8

#
def set_seed(seed_value=42):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model_yaml_path = "ultralytics/cfg/models/v10/yolov10.yaml"

data_yaml_path = r'/root/yolov10-main/ultralytics/cfg/datasets/rebar.yaml'

if __name__ == '__main__':

    seed_value = 42  # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    set_seed(seed_value)

    model = YOLOv10(model_yaml_path)

    results = model.train(
        data=data_yaml_path,
        epochs=300,
        batch=8,
        name='train_v10',
        seed=seed_value,
        deterministic=True,
    )
