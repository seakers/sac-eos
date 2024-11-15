import torch

a = torch.FloatTensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

print(a)

print(a.view(-1))

a = a.view(-1)

b = torch.cat([a, torch.FloatTensor([0 for _ in range(3)])])

print(b)

c = torch.zeros(3)

print(c)

print(c.shape[0])