import torch
import torch.optim as optim
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)


optimizer = optim.SGD(w1.parameters(), lr = 0.01)
d = torch.mean(w1)
d.backward()
print(w1.grad)
