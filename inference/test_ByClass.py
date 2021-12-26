import torch
from transforms_squarepad3 import categories
from train_sqaurepad import val_loader,device, criterion, model
import wandb
import numpy as np

"""### Inference"""
PATH = '/workspace/pytorch/project_dir/Ratio_Image_Recognition/ResNet50_SquarePad_lr0.00001.pt'

model = model.to(device)
#print(device)

batch_size = 1
model.load_state_dict(torch.load(PATH))

model.eval()

test_loss = 0.0

correct = 0
total = 0

class_correct = list(0. for i in range(13))
class_total = list(0. for i in range(13))

example_images = []
with torch.no_grad():

    for data, target in val_loader:
        if len(target.data) != batch_size:
            break

        data = data.to(device)
        target = target.to(device)

        # forward pass: 입력을 모델로 전달하여 예측된 출력 계산
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # 출력된 확률을 예측된 클래스로 변환
        _, pred = torch.max(output, 1)
        # 예측과 실제 라벨과 비교
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # 각 object class에 대해 test accuracy 계산
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct.item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(val_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(13):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (categories[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


'''
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, squarepad_visual)
        c = (predicted == labels).squeeze()

        example_images.append(wandb.Image(
            data[0], caption = "Pred : {}, Truth : {}".format(predicted[0].item, labels[0])
        ))

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(4):
            c_labels = labels
            class_correct[c_labels] += c.item()
            class_total[c_labels] += squarepad_visual

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


wandb.log({
    "Examples": example_images,
    "Test Accuracy": 100. * correct / len(testloader.dataset),
    })

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        categories[i], 100 * class_correct[i] / class_total[i]))
'''

