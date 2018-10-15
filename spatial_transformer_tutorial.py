from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import kit
import models


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}    Loss: {:.4f}'.format(epoch, loss.item()))


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Acc: {:.0f}%'
              .format(test_loss, 100. * correct / len(test_loader.dataset)))


def visualize_stn():
    with torch.no_grad():
        # [64,1,28,28]

        fsize = 30
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = kit.convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = kit.convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Testset Images', fontsize=fsize)

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images', fontsize=fsize)

    with torch.no_grad():
        # [64,1,28,28]
        data = next(iter(train_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = kit.convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = kit.convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Trainset Images', fontsize=fsize)

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images', fontsize=fsize)


def get_modified_mnist():
    train_set = datasets.MNIST(root='.', train=True, download=True, transform=transforms.Compose([
        models.DoubleGauss(),
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_set = datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        models.DoubleGauss(),
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
    # kit.plot_data(next(iter(test_loader))[0])
    return train_loader, test_loader


if __name__ == '__main__':

    plt.ion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, test_loader = get_modified_mnist()
    model = models.Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, 20 + 1):
        t1, t2 = time.time(), time.clock()
        train(epoch)
        t3, t4 = time.time(), time.clock()
        # print(f'time:%.2f,clock:%.2f for one training epoch.' % (t3 - t1, t4 - t2))
        test()

    visualize_stn()
    plt.ioff()
    plt.show()

    a = 1
