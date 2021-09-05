import torchvision.transforms.functional as TF
import numpy as np
import torch.nn.functional as F


class SquarePad:
   def __call__(self, image):
      w, h = image.size
      max_wh = np.max([w, h])
      hp = int((max_wh - w) / 2)
      vp = int((max_wh - h) / 2)
      padding = (hp, vp, hp, vp)
      return TF.pad(image, padding, 0, 'constant')

# 랜덤으로 적용
# 데이터를 더해줌(데이터 증강)


'''
# usage

UnSquarePad.Compose(
        [
            SquarePad(),
            UnSquarePad.Resize([224, 224]),
            UnSquarePad.ToTensor(),
        ]
)
'''