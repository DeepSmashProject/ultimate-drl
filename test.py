import torch

a = torch.zeros(10)
damage = torch.ones(2)
print(a.size(), a, damage)
print(torch.cat([a, damage]))
b = torch.cat([a, damage])
print(b.size(), a, damage)