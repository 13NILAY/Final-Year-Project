import torch
x = torch.randn(5000, 5000, device="cuda")
y = torch.matmul(x, x)
print("Done on:", y.device)