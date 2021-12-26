#from transforms_squarepad import categories, valset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from Args import *

'''
model = models.densenet201(pretrained = True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=Args["batch_size"], shuffle=False, num_workers = 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = '/workspace/pytorch/project_dir/Ratio_Image_Recognition/weights/SquarePad1.pt'

model = model.to(device)
model.cuda()


nb_classes = 1000
confusion_matrix = np.zeros((nb_classes, nb_classes))
model.load_state_dict(torch.load(PATH))

model.eval()

with torch.no_grad():
    for i, (inputs, classes) in enumerate(val_loader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


plt.figure(figsize=(25,20))
print(confusion_matrix)

class_names = list(categories)
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
'''
'''
df_cm = [
    [72.06,	10.87,	4.32,	0.21,	5.98,	7.28,	1.84,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00],
    [15.43,	70.24,	9.74,	0.20,	0.00,	0.06,	6.58,	3.95,	0.21,	0.35,	0.00,	3.42,	0.00],
    [5.15,	12.71,	80.51,	0.00,	0.00,	0.00,	1.81,	4.79,	0.32,	3.58,	2.41,	0.00,	1.78],
    [0.00,	0.00,	0.00,	78.49,	17.17,	0.08,	4.23,	0.36,	11.14,	0.00,	0.00,	0.00,	0.00],
    [1.02,	0.00,	0.00,	16.59,	68.92,	2.78,	0.00,	0.09,	3.78,	0.67,	0.00,	0.00,	0.00],
    [3.88,	0.00,	0.00,	0.08,	3.48,	73.22,	6.79,	0.51,	0.00,	0.00,	0.00,	0.00,	0.19],
    [0.11,	3.58,	1.10,	0.00,	0.00,	11.15,	67.74,	10.44,	0.00,	0.00,	0.00,	0.00,	1.57],
    [2.22,	1.75,	4.01,	0.00,	0.00,	4.82, 10.80,	78.54,	0.87,	0.00,	0.00,	0.00,	0.00],
    [0.00,	0.00,	0.00,	3.56,	4.45,	0.61,	0.00,	0.00,	71.86,	8.82,	0.00,	0.00,	0.00],
    [0.13,	0.00,	0.00,	0.20,	0.00,	0.00,	0.21,	0.00,	11.32,	67.31,	4.28,	8.46,	0.00],
    [0.00,	0.85,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	6.03,	78.74,	8.93,	5.40],
    [0.00,	0.00,	0.32,	0.67,	0.00,	0.00,	0.00,	1.32,	0.00,	8.34,	14.29,	68.34,	7.48],
    [0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.50,	4.90,	0.28, 10.85,	83.58]

]
'''
'''
df_cm = [[74.08, 16.76,	3.60,	0.00,	2.13,	2.45,	0.14,	0.00,	0.00,	0.58,	0.26,	0.00,	0.00],
         [12.73,73.66,	8.20,	0.00,	0.00,	0.32,	2.34,	0.18,	0.00,	0.00,	0.85,	1.14,	0.58],
         [0.00,	12.57,	84.69,	0.00,	0.00,	0.00,	0.00,	2.44,	0.00,	0.00,	0.00,	0.00,	0.30],
         [0.18,	0.20,	0.00,	80.00,	15.93,	0.04,	0.08,	0.05,	2.38,	0.40,	0.06,	0.64,	0.00],
         [4.82,	0.00,	0.00,	15.21,	74.92,	2.45,	0.00,	0.00,	3.41,	0.00,	0.00,	0.03,	0.00],
         [6.76,	0.09,	0.00,	0.16, 2.81,	75.84,	10.86,	2.30,	0.94,	0.00,	0.00,	0.24,	0.00],
         [1.82,	5.60,	1.81,	1.26,	0.00,	4.79, 73.84,	9.73,	0.00,	0.34,	0.00,	0.81,	0.00],
         [0.00,	1.94,	6.89,	0.48,	0.36,	0.67,	7.93,	78.74,	0.00,	0.00,	0.00,	1.83,	1.16],
         [0.00,	0.19,	0.31,	10.23,	3.53,	0.00,	0.00,	0.81,	73.32,	10.59,	1.02,	0.00,	0.00],
         [0.00,	0.55,	4.79,	0.00,	0.64,	0.00,	0.00,	0.00,	9.36,	67.74,	5.15,	7.65,	4.12],
         [0.00,	0.00,	2.36,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	4.51,	80.96,	12.17,	0.00],
         [0.00,	3.14,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	7.88,	8.86,	71.33,	8.79],
         [0.00,	0.00,	1.65,	0.00,	0.00,	0.14,	1.74,	0.00,	0.00,	0.00,	5.74,	6.10,	84.63]

]
'''

df_cm = [[73.55,15.32,	3.40,	0.00,	2.21,	2.53,	0.12,	0.00,	0.00,	0.62,	0.25,	0.00,	0.00],
         [11.78,73.37,	8.39,	0.00,	0.00,	0.33,	2.39,	0.17,	0.00,	0.00,	0.82,	1.16,	0.59],
         [0.00,	11.76,	82.93,	0.59,	0.00,	0.21,	0.00,	3.86,	0.00,	0.00,	0.00,	0.00,	0.65],
         [0.16,	0.27,	0.00,	81.28,	13.97,	0.43,	0.08,	0.05,	2.53,	0.50,	0.08,	0.65,	0.00],
         [5.81,	0.00,	0.00,	12.18,	72.96,	2.48,	0.00,	0.00,	5.08,	0.00,	0.00,	0.49,	0.00],
         [5.54,	0.09,	0.00,	0.18,	2.81,	76.82,	10.43,	3.13,	0.86,	0.00,	0.00,	0.14,	0.00],
         [1.94,	6.08,	1.81,	1.56,	0.00,	4.63,	74.67,	7.89,	0.00,	0.58,	0.00,	0.84,	0.00],
         [0.00,	2.68,	5.49,	0.57,	0.68,	0.69,	7.88,	78.73,	0.00,	0.00,	0.00,	1.96,	1.32],
         [0.00,	0.19,	0.31,	10.21,	3.55,	0.00,	0.00,	0.81,	72.89,	10.86,	1.18,	0.00,	0.00],
         [0.00,	0.85,	5.29,	0.32,	0.78,	0.00,	0.00,	0.00,	7.84,	66.86,	6.52,	7.38,	4.16],
         [0.00,	0.24,	2.63,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	4.68,	79.96,	12.49,	0.00],
         [0.00,	3.22,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	7.91,	8.83,	71.35,	8.69],
         [0.00,	0.00,	1.63,	0.00,	0.00,	0.12,	1.79,	0.00,	0.00,	1.42,	5.68,	5.88,	83.48],

]

labels = ["short skirts",	"midi skirts", "long skirts",	"top",	"crop tee",	"short pant", "midi pants", 'long pants','shirts','shirts dress', 'mini dress','midi dress','long dress']

#from sklearn.metrics import recall_score

#print(recall_score(labels,df_cm))

plt.figure(figsize=(14,12))
heatmap = sns.heatmap(df_cm, xticklabels=labels, yticklabels=labels, annot=True, fmt="")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=14)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()
