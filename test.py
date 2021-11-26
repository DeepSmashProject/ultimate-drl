import torch

a = torch.zeros(32, 10)
damage = torch.ones(32, 2)
print(a.size(), damage.size())
print(torch.cat([a, damage], dim=1))
b = torch.cat([a, damage], dim=1)
print(b.size(), a, damage)