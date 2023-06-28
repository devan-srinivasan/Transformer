"""
This is straight from pytorch tutorial lol their is nothing really here 
"""
import torch

def tut():
    from torchvision.models import resnet18, ResNet18_Weights
    # download model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # fake training data & labels
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)
    # forward pass
    prediction = model(data)
    # compute loss
    loss = (prediction - labels).sum()
    # backwards pass
    loss.backward()
    # stochastic gradient descent :D
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()
    # now the model has learned one step, loop this badboy with real data and you got yourself a learning model

# Q, K, V = torch.randint(1, 2, (1, 2, 3)), torch.randint(1, 2, (1, 2, 3)), torch.randint(1, 2, (1, 2, 3))
# print(torch.bmm(torch.nn.functional.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)), V.float()))
# x = torch.ones(4, 4)
# y = torch.arange(4)
# print(torch.where(y % 2 == 0, 2*x, 3*x))

def f(*args):
    print(args[0])
    print(*args)

f(1, 2, 3, 4, 5)