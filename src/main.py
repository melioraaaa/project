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
