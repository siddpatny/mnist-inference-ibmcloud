import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import io
from PIL import Image
import flask
from flask import request
import time

app = flask.Flask(__name__)

port = int(os.getenv("PORT"))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    #image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image_bytes).unsqueeze(0)

def train(model, device, train_loader, optimizer, epoch,log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


@app.route('/inference', methods=['POST'])
def test():
    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt', map_location='cpu')) 
    # model.eval()
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            test_image = flask.request.files["image"].read()
            test_image = Image.open(io.BytesIO(test_image))
    tensor = transform_image(image_bytes=test_image)
    output = model(tensor)
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    return flask.jsonify({'class_name': pred.item()})


@app.route('/train', methods=['POST'])
def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=5, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')

    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()
    start_time = time.monotonic()

    use_cuda = torch.cuda.is_available()

    batch_size = request.args.get('batch-size',default=8,type=int)
    epochs = request.args.get('epochs',default=10,type=int)
    lr = request.args.get('lr', default = 0.3, type = float)
    gamma = request.args.get('gamma', default = 0.7, type = float)
    log_interval = request.args.get('log', default = 100, type = int)
    

    torch.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch,log_interval)
        # test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
    end_time = time.monotonic()
    return flask.jsonify({'total time': (end_time - start_time)})

    # class_name = test(model,device)
    # return flask.jsonify({'class_name': class_name})


if __name__ == '__main__':
    # main()
    app.run(host='0.0.0.0', port=8001)