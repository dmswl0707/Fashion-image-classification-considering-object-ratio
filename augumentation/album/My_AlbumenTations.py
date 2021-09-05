import albumentations
from albumentations.pytorch.transforms import ToTensorV2
#from functions.squarepad import *
from Albums import *
from Args import *

import cv2
import time
import os
import imageio
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms


albumentations_train = albumentations.Compose([
                                 albumentations.SquarePad(), #square pad 적용
                                 albumentations.Resize(224, 224),
                                 albumentations.RandomRotate90(p=0.5),
                                 albumentations.HorizontalFlip(),
                                 #albumentations.pytorch.UnSquarePad.ToTensorV2(),
                                 albumentations.Normalize(Args["mean"], Args["std"], max_pixel_value=255.0),
                                 albumentations.pytorch.transforms.ToTensorV2(),
])

albumentations_val = albumentations.Compose([
                          albumentations.SquarePad(),  # square pad 적용
                          albumentations.Resize(224, 224),
                          albumentations.RandomRotate90(p=0.5),
                          albumentations.HorizontalFlip(),
                          #albumentations.pytorch.UnSquarePad.ToTensorV2(),
                          albumentations.Normalize(Args["mean"], Args["std"]),
                          albumentations.pytorch.transforms.ToTensorV2(),
])

'''
file_paths='/workspace/pytorch/dataset/train/*/*.JPEG'
print(file_paths)

f=glob(file_paths)
#print(f)
for i in range(0, 3):
    file = f[i]
    #format = "'{0}'".format(file)
    print(file)
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('sample', image)
    cv2.waitKey()
    cv2.destroyAllWindows()®®
'''

albumentations_trainset = AlbumentationsDataset(
    file_paths=["/workspace/pytorch/dataset/train/n03393912/n03393912_6524.JPEG"],
    labels=[1],
    transform = albumentations_train
)

albumentations_valset = AlbumentationsDataset(
    file_paths=["/workspace/pytorch/dataset/train/n03393912/n03393912_6524.JPEG"],
    labels=[1],
    transform = albumentations_val
)

##
for i in range(100):
  sample, _, transform_time = albumentations_trainset[0]

plt.imshow(transforms.ToPILImage()(sample))
plt.show()

'''
for i in range(num_samples):
  print(albumentations_trainset)
  print(albumentations_trainset[0])
  ax[i].imshow(albumentations_trainset[0])
  ax[i].axis('off')
  #plt.imshow((out * 255).astype(np.uint8))'''

'''
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=Args["batch_size"], shuffle = True, num_workers = 8)
#val_loader = torch.utils.data.DataLoader(valset, batch_size=Args["batch_size"], shuffle=False, num_workers = 8)

#categories = list(trainset.class_to_idx.keys())

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
'''