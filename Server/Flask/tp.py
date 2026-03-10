import torch
x = torch.randn(5000, 5000, device="cpu")
y = torch.matmul(x, x)
print("Done on:", y.device)