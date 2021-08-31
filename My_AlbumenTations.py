import albumentations
from albumentations.pytorch.transforms import ToTensorV2
#from torchvision.datasets import ImageFolder
#from torchvision import transforms
from functions.squarepad import *
from Args import *
import matplotlib.pyplot as plt

import cv2
import time
import os
import imageio
from glob import glob


class Albumentation_Dataset():

    def __init__(self, data_list, labels, transform):
        #self.file_path = file_path
        #self.img_set = img_set
        #self.label_file = label_file
        self.path_list = data_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.path_list)
    '''
    def get_label(self, idx):
        label = self.labels[idx]
        label_list = []
    '''

    def __getitem__(self, idx):
        label = self.labels
        #file_path = self.file_path[idx]
        #img_set = self.img_set
        #img_path = self.file_path + self.img_set
        '''
        if self.img_set == "train":
            train = self.file_path + '/' + 'train'
            img_path = train

        elif self.img_set == "val":
            val = self.file_path + '/' + 'val'
            img_path = val
        '''

        #img_list = glob(file_path + '/*/*.JPEG')
        #img_list = cv2.imread(img_list)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = imageio.imread(self.path_list[idx])
        #image = cv2.cvtColor(img_list, cv2.COLOR_BGR2RGB)

        #os.chdir(img_path)
        #img_list = cv2.imread(img_list)
        #img_data = np.asarray(img_list)
        #image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        time_start = time.time()

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented[image]
            total_time = (time.time() - time_start)

        return image, label, total_time


albumentations_train = albumentations.Compose([
                                 SquarePad(), #square pad 적용
                                 albumentations.Resize(224, 224),
                                 #albumentations.RandomRotate90(p=0.5),
                                 albumentations.HorizontalFlip,
                                 albumentations.pytorch.transforms.ToTensorV2(),
                                 albumentations.Normalize(Args["mean"], Args["std"])
])

albumentations_val = albumentations.Compose([
                          SquarePad(),  # square pad 적용
                          albumentations.Resize(224, 224),
                          #albumentations.RandomRotate90(p=0.5),
                          albumentations.HorizontalFlip,
                          albumentations.pytorch.transforms.ToTensorV2(),
                          albumentations.Normalize(Args["mean"], Args["std"])
])

train_list = glob('/workspace/pytorch/dataset/train/*/*.JPEG')
val_list = glob('/workspace/pytorch/dataset/val/*/*.JPEG')

albumentations_trainset = Albumentation_Dataset(
    data_list=train_list,
    #img_set=['train'],
    labels=[1],
    transform = albumentations_train
)

albumentations_valset = Albumentation_Dataset(
    data_list=val_list,
    #img_set=['val'],
    labels=[1],
    transform = albumentations_val
)

num_samples = 5
fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
for i in range(num_samples):
  ax[i].imshow(albumentations_trainset[0][0])
  ax[i].axis('off')
