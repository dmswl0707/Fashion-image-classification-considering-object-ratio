from torchvision import transforms
#from functions.squarepad_visual import *
from Args2 import *
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder

# imgnet_preprocessing (이미지 로드 확인용)
# 데이터 증강과 데이터 로드 #이미지넷


transforms_train = transforms.Compose([
                                 #SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 transforms.RandomRotation(degrees=20),
                                 transforms.RandomHorizontalFlip(),
                                 #transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                 transforms.RandomPosterize(2, p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                ])

transforms_val = transforms.Compose([
                                 #SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #UnSquarePad.RandomRotation(degrees=20),
                                 #UnSquarePad.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                 ])

transforms_test = transforms.Compose([
                                 #SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #UnSquarePad.RandomRotation(degrees=20),
                                 #UnSquarePad.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(Args["mean"], Args["std"])
                                 ])

# 이미지 넷 불러오기

trainset = ImageFolder(root='/workspace/DataSet/train', transform= transforms_train)
valset = ImageFolder(root='/workspace/DataSet/val', transform = transforms_val)
testset = ImageFolder(root='/workspace/DataSet/val', transform = transforms_test)


num_train = len(trainset)
num_val = len(valset)
print({'train' : num_train})
print({'val' : num_val})


categories = list(trainset.class_to_idx.keys())
#print(categories)
num_class = len(categories)
#print(num_class)

#train_loader = torch.utils.data.DataLoader(trainset, batch_size=Args["batch_size"], shuffle = True, num_workers = 4)
#val_loader = torch.utils.data.DataLoader(valset, batch_size=Args["batch_size"], shuffle=False, num_workers = 4)

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
  ax = fig.add_subplot(2, 10, i+squarepad_visual, xticks=[], yticks=[])
  plt.imshow(im_convert(images[i]))
  ax.set_title(categories[labels[i].item()])
  plt.show()

'''