

from transforms import *
from training import model, criterion, optimizer


def check_batch(model, train_loader, criterion, optimizer):

    images, labels = next(iter(train_loader))

    for i in range(30):
        data = images.to(Args["device"])
        target = labels.to(Args["device"])

        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


#check_batch(model, train_loader, criterion, optimizer)