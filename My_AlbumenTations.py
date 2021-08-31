import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.datasets import ImageFolder
#from torchvision import transforms
from functions.squarepad import *
from Args import *
import matplotlib.pyplot as plt

import cv2
import time
import os
import imageio
from glob import glob

'''
class Albumentation_Dataset():

    def __init__(self, file_path, labels, transform):
        self.file_path = file_path
        #self.img_set = img_set
        #self.label_file = label_file
        #self.path_list = data_list
        self.file_path = file_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_path)
    
    def get_label(self, idx):
        label = self.labels[idx]
        label_list = []

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_path[idx]
        #img_set = self.img_set
        #img_path = self.file_path + self.img_set
        
        if self.img_set == "train":
            train = self.file_path + '/' + 'train'
            img_path = train

        elif self.img_set == "val":
            val = self.file_path + '/' + 'val'
            img_path = val
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = imageio.imread(self.path_list[idx])
        #image = cv2.cvtColor(img_list, cv2.COLOR_BGR2RGB)
        
        #os.chdir(img_path)
        img_list = cv2.imread(file_path)
        img_data = np.asarray(img_list)
        image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        time_start = time.time()

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented[image]
            total_time = (time.time() - time_start)

        return image, label, total_time
'''

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

#train_list = glob('/workspace/pytorch/dataset/train/*/*.JPEG')
#val_list = glob('/workspace/pytorch/dataset/val/*/*.JPEG')
'''
albumentations_trainset = Albumentation_Dataset(
    file_path=['/workspace/pytorch/dataset/train/*/*.JPEG'],
    #img_set=['train'],
    labels=[1],
    transform = albumentations_train
)

albumentations_valset = Albumentation_Dataset(
    file_path=['/workspace/pytorch/dataset/val/*/*.JPEG'],
    #img_set=['val'],
    labels=[1],
    transform = albumentations_val
)

num_samples = 5
fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
for i in range(num_samples):
  ax[i].imshow(albumentations_trainset[0][0])
  ax[i].axis('off')
'''
trainset = ImageFolder(root='/workspace/pytorch/dataset/train', transform= albumentations_train)
valset = ImageFolder(root='/workspace/pytorch/dataset/val', transform = albumentations_val)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Args["batch_size"], shuffle = True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=Args["batch_size"], shuffle=False, num_workers = 8)

categories = list(trainset.class_to_idx.keys())

def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  image = image.clip(0, 1)
  return image


dataiter = iter(train_loader)
images, labels = dataiter.next()

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
  # row 2 column 10
  ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[i]))
  ax.set_title(categories[labels[i].item()])
  plt.show()
