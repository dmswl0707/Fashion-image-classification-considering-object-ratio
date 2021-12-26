import torch.optim as optim
#import torch.optim.lr_scheduler
import torch.optim.lr_scheduler
import torchvision.models as models
from functions.early_stopping2 import *

from transforms_unsquarepad3 import *
from functions.custom_CosineAnnealingWarmupRestart import *
from Args2 import *
import wandb

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Args["batch_size"],  sampler = train_sampler, num_workers = 8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=Args["batch_size"], sampler = val_sampler, num_workers = 8)
#test_loader = torch.utils.data.DataLoader(testset, batch_size=squarepad_visual, shuffle=False, num_workers = 8)
#testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=8)

# 스크래치 학습 ㄴㄴ 미세조정 학습ㄷ
model = models.resnet50(pretrained = True)

if torch.cuda.device_count()>1:
    net = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")


wandb.init(name = Args["name"])

# train setting
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


network = net.to(device)
print(device)

# 하이퍼 파라미터 변경
optimizer = optim.Adam(model.parameters(), lr=Args["lr"], betas=(0.9, 0.999), eps=1e-08)
#scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=1, eta_max=Args["eta_min"], T_up=3, gamma=0.3)
Epoch = Args["Epoch"]
patience = Args["patience"]

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.5, gamma=0.3)

wandb.watch(network)


def training():

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(Epoch):
        print("===================================================")
        print("epoch: ", epoch + 1)

        train_loss, val_loss = [], []
        avg_train_loss, avg_val_loss =  [], []

        total = 0
        v_total = 0

        train_loss = 0.0
        train_correct = 0.0
        val_loss = 0.0
        val_correct = 0.0

        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            X = inputs.to(device)
            y = labels.to(device)

            y_pred = network(X)
            loss = criterion(y_pred, y)

            loss.backward()
            # step 마다 돌아가는 부분
            optimizer.step()

            # acc를 구하는 부분
            _, preds = torch.max(y_pred, 1)

            total += y.size(0)
            train_loss += loss.item()
            #train_loss.append(loss.item())
            train_correct += (preds == y).sum().item()

            np_train_loss = np.average(train_loss)
            np_train_acc = np.average(100. * float(train_correct) / total)

        epoch_loss = np_train_loss / len(train_loader)
        epoch_acc = np_train_acc

        print("train loss: {:.4f}, acc: {:4f}".format(epoch_loss, epoch_acc))

        wandb.log({
            "Train Loss": epoch_loss,
            "custom_epoch" : epoch,
            "Train Accuracy": epoch_acc,
            "Train error": 100 - epoch_acc,
            "lr" : optimizer.param_groups[0]['lr'] # 학습률 로깅
        })

        with torch.no_grad():

            model.eval()
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                y_val_pred = network(val_inputs)
                v_loss = criterion(y_val_pred, val_labels)

                #validation - acc를 구하는 부분
                _, v_preds = torch.max(y_val_pred, 1)

                v_total += val_labels.size(0)
                val_loss += v_loss.item()
                #val_loss.append(v_loss.item())
                val_correct += (v_preds == val_labels).sum().item()

                np_val_loss = np.average(val_loss)
                np_val_acc = np.average(100. * float(val_correct) / v_total)

            val_epoch_loss = np_val_loss / len(val_loader)
            val_epoch_acc = np_val_acc

            print("val loss: {:.4f}, acc: {:4f}".format(val_epoch_loss, val_epoch_acc))

            wandb.log({
                "Val Loss": val_epoch_loss,
                "Val Accuracy": val_epoch_acc,
                "Val error": 100 - val_epoch_acc,
            })

            scheduler.step()

            early_stopping(val_epoch_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
