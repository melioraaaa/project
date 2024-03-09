#1 блок 
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())

larger_than_limit_sum = x[x > limit].sum()

print(larger_than_limit_sum)

#2 блок.
#задача 1
import torch

w = torch.tensor([[5, 10], [1, 2]], dtype=torch.float, requires_grad=True)

function =  torch.prod (torch.log(torch.log(w+7)))
function.backward()

print(w.grad)
#Задача 2
import torch

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001

for _ in range(500):
    function = (w + 7).log().log().prod()
    function.backward()
    w.data -= alpha * w.grad 
    w.grad.zero_()
print(w)
#Задача 3
import torch

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
optimizer =  torch.optim.SGD([w],lr = 0.001)

for _ in range(500):
    function = (w + 7).log().log().prod()
    function.backward()
    optimizer.step()
    optimizer.zero_grad()
print(w)
# Задача 4
import torch

class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

sine_net = SineNet(int(input()))

output = sine_net.forward(torch.Tensor([1.]))

print(sine_net)
