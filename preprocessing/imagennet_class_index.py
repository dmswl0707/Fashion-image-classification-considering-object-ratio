import json
import os

idx2label = []
cls2label = {}
with open("/workspace/pytorch/project_dir/Ratio_Image_Recognition/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

#print(cls2label)
#print(idx2label)

train_path = '/workspace/pytorch/dataset/train'
val_path = '/workspace/pytorch/dataset/val'

train_folder_name = os.listdir(train_path) # train_set, 딕셔너리 key 값
#print(train_folder_name)
#print(len(train_folder_name))
val_folder_name = os.listdir(val_path)

key = list(cls2label.keys())
value = list(cls2label.values())

#print(key)
#print(len(key))
#print(value)
#print(len(value))

'''
# train set 폴더 이름 바꿔주기

for i in range(0, len(train_folder_name)):

    for key[i] in key:
        #print(key[i])
        src = os.path.join(train_path, key[i])
        #print(src)

    # 새로 바꿔주는 이름
    for value[i] in value:
        #print(value[i])
        dsts = os.path.join(train_path, value[i])
        #print(dsts)

    os.rename(src, dsts)


# val set 폴더 이름 바꿔주기

for i in range(0, len(val_folder_name)):

    for key[i] in key:
        #print(key[i])
        src = os.path.join(val_path, key[i])
        #print(src)

    # 새로 바꿔주는 이름
    for value[i] in value:
        #print(value[i])
        dsts = os.path.join(val_path, value[i])
        #print(dsts)

    os.rename(src, dsts)
'''

