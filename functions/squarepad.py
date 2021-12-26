import torch as torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from PIL import Image
import glob
from Args import *

class SquarePad:
   def __call__(self, image):
      w, h = image.size
      max_wh = np.max([w, h])
      hp = int((max_wh - w) / 2)
      vp = int((max_wh - h) / 2)
      padding = (hp, vp, hp, vp)
      return TF.pad(image, padding, 0, 'constant')


image_path= '/workspace/pytorch/project_dir/Ratio_Image_Recognition/functions/squarepad_visual'
'''
transforms_ = transforms.Compose([
                                 SquarePad(), #square pad 적용
                                 transforms.Resize((224, 224)),
                                 #UnSquarePad.RandomRotation(degrees=20),
                                 #UnSquarePad.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 #transforms.ToPILImage(),®
                                 transforms.Normalize(Args["mean"], Args["std"])
                                 ])

testset = ImageFolder(root=image_path, transform = transforms_)
Testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)


def custom_imshow(img):

    img = img.numpy() #RGB로 바꾸어줘라
    #cv2.imread('')
    #numpy에서 이미지로 받아오기
    #bgr = cv2.imread(img)
    #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)), )
    plt.show()

def process():
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        custom_imshow(inputs[0])

process()

'''