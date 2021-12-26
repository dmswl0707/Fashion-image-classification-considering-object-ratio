import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from train_sqaurepad import model, device
import time
from train_sqaurepad import testloader

def get_clf_eval(y_true, y_pred, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    return accuracy, precision, recall, f1


PATH = '/workspace/pytorch/project_dir/Ratio_Image_Recognition/DenseNet201_SquarePad_lr3e-6.pt'
model = model.to(device)
model.load_state_dict(torch.load(PATH))

test_start = time.time()

model.eval()
with torch.no_grad():
    test_acc_tmp, test_precision_tmp, test_recall_tmp, test_f1_tmp = [], [], [], []
    for test_iter, (test_x, test_y_true) in enumerate(testloader):
        test_x, test_y_true = test_x.to(device), test_y_true.to(device)
        test_y_pred = model.forward(test_x) # forward

        _, test_pred_index = torch.max(test_y_pred, 1)
        test_pred_index_cpu = test_pred_index.cpu().detach().numpy()
        test_y_true_cpu = test_y_true.cpu().detach().numpy()
        test_acc, test_precision, test_recall, test_f1 = get_clf_eval(test_y_true_cpu, test_pred_index_cpu)
        test_acc_tmp.append(test_acc), test_precision_tmp.append(test_precision), test_recall_tmp.append(test_recall), test_f1_tmp.append(test_f1)

    test_acc_mean = sum(test_acc_tmp, 0.0)/len(test_acc_tmp)
    test_precision_mean = sum(test_precision_tmp, 0.0)/len(test_precision_tmp)
    test_recall_mean = sum(test_recall_tmp, 0.0)/len(test_recall_tmp)
    test_f1_mean = sum(test_f1_tmp, 0.0)/len(test_f1_tmp)
    print("[Evaluation] {:.2f}[s], Test Accuracy : {:.4f}, Precision : {:.4f}, Recall : {:.4f}, F1 Score : {:.4f}".format( time.time()-test_start, test_acc_mean, test_precision_mean, test_recall_mean, test_f1_mean))
    print("[Model Performance] Model Performance : {:.5f}".format(test_acc_mean))

