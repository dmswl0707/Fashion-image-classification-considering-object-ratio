from torchvision import transforms
from functions.squarepad import *
from Args import *
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder

# preprocessing (이미지 로드 확인용)
# 데이터 증강과 데이터 로드


transforms_train = transforms.Compose([
                                 SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #transforms.RandomRotation(degrees=30),
                                 transforms.RandomHorizontalFlip(),\
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                ])

transforms_val = transforms.Compose([
                                 SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #transforms.RandomRotation(degrees=20),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                 ])

# 이미지 넷 불러오기

trainset = ImageFolder(root='/workspace/pytorch/dataset/train', transform= transforms_train)
valset = ImageFolder(root='/workspace/pytorch/dataset/val', transform = transforms_val)

num_train = len(trainset)
num_val = len(valset)
print({'train' : num_train})
print({'val' : num_val})


categories = list(trainset.class_to_idx.keys())
#print(categories)
num_class = len(categories)
#print(num_class)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Args["batch_size"], shuffle = True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=Args["batch_size"], shuffle=False, num_workers = 8)

### 이미지 확인
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

