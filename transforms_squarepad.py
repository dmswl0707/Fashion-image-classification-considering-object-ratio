import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from functions.squarepad import *
from preprocessing.imagennet_class_index import value
from Args import *
import matplotlib.pyplot as plt
import numpy as np


transforms_train = transforms.Compose([
                                 SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 transforms.RandomRotation(degrees=20),
                                 transforms.RandomHorizontalFlip(),
                                 #transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                 transforms.RandomPosterize(2, p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                ])

transforms_val = transforms.Compose([
                                 SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #UnSquarePad.RandomRotation(degrees=20),
                                 #UnSquarePad.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                 ])

transforms_test = transforms.Compose([
                                 SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #UnSquarePad.RandomRotation(degrees=20),
                                 #UnSquarePad.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                 ])

trainset = ImageFolder(root='/workspace/pytorch/dataset/train', transform= transforms_train)
valset = ImageFolder(root='/workspace/pytorch/dataset/val', transform = transforms_val)
testset = ImageFolder(root='/workspace/pytorch/dataset/val', transform = transforms_test)


num_train = len(trainset)
num_val = len(valset)
#num_test = len(testset)

print({'train' : num_train})
print({'val' : num_val})
#print({'test' : num_test})

categories = value
#categories = list(trainset.class_to_idx.keys())
#print(categories)
num_class = len(categories)
#print(num_class)

### 이미지 확인
def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  image = image.clip(0, 1)
  return image

'''
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
