import json
import os


idx2label = []
cls2label = {}
with open("/workspace/pytorch/project_dir/Ratio_Image_Recognition/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}


#print(cls2label) # Key 값 불러서  Value로 이름 바꾸기

train_path = '/workspace/pytorch/dataset/train'
val_path = '/workspace/pytorch/val'

train_folder = os.listdir(train_path)
#print(train_folder)

# train set 폴더 이름 바꿔주기
for dir in train_folder:
    src = os.path.join(train_folder, dir)
    #딕셔너리 키 접근하여 value로 이름 변경
    dst =
    dst = os.path.join(train_folder, dst)
    os.rename(src, dst)

# Val set 폴더 이름 바꿔주기
for dir in train_folder:
    src = os.path.join(train_folder, dir)
    #딕셔너리 키 접근하여 value로 이름 변경
    dst =
    dst = os.path.join(train_folder, dst)
    os.rename(src, dst)

