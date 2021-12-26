import torch
import torch.nn as nn


Args = {"name" : 'ResNet50_UnSquarePad_lr0.00001',
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "lr" : 0.000001,  # 스크래치는 논문 그대로 , 전이학습은 squarepad_visual/10로...
        #"eta_min" : 0.000005,
        #"weight_decay" : 0.001,
        "batch_size": 80, # 2 gpus 80, squarepad_visual gpu 32
        "Epoch" : 20,
        #"num_fold" : 5,
        "patience" : 20,
        "device" : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        }


