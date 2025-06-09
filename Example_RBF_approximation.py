import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from better_kan import KAN
class RBF(nn.Module):
    def __init__(self, in_features, num_centers, sigma):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.centers, -1, 1)  # Assuming input range [-1,1]

    def forward(self, inputs):
        size = (inputs.size(0), self.num_centers, self.in_features)
        x_expanded = inputs.unsqueeze(1).expand(size)
        centers_expanded = self.centers.unsqueeze(0).expand(size)
        return torch.exp(-torch.norm(x_expanded - centers_expanded, dim=2) ** 2 / (2 * self.sigma ** 2))

class RBFNetwork(nn.Module):
    def __init__(self, in_features, num_centers, out_features, sigma):
        super(RBFNetwork, self).__init__()
        self.rbf = RBF(in_features, num_centers, sigma)
        self.linear = nn.Linear(num_centers, out_features)

    def forward(self, x):
        rbf_basis = self.rbf(x)
        # print(rbf.size(),self.linear.weight.size())
        rbf=rbf_basis.unsqueeze(1)*self.linear.weight.unsqueeze(0)
        rbf=rbf.squeeze(1)
        out = self.linear(rbf_basis)
        return out,rbf

# Model and optimizer setup
num_centers = 100
sigma = 0.1
model = RBFNetwork(in_features=1, num_centers=num_centers, out_features=1, sigma=sigma)
optimizer = optim.LBFGS(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Data Generation for Bessel function J0(20x)
x = torch.linspace(-2, 2, steps=1000).unsqueeze(1)
y = torch.special.bessel_j0(20 * x)

# Training loop without batching
def train(model, optimizer, loss_fn, epochs, x, y):
    model.train()
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            output = model(x)[0]
            loss = loss_fn(output, y)
            loss.backward()
            return loss
        loss=optimizer.step(closure)
        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train(model, optimizer, loss_fn, epochs=500, x=x, y=y)

# Plotting the results
plt.figure(figsize=(10, 6))
predicted,rbf = model(x)
predicted,rbf=predicted.detach(),rbf.detach()
for i in range( rbf.size(1)):
    plt.plot(x.numpy(), rbf[:,i].numpy(), label=f'Gaussian Kernel {i}',color='black')

plt.plot(x.numpy(), predicted.numpy(), label='RBF Model Prediction', linestyle='--',color='red')
# plt.plot(x.numpy(), y.numpy(), label='True Function',color='blue')
plt.legend()
plt.title('Radial Basis Functions Approximation for bessel_j0(20*x) in range [-2,2]')
plt.xlabel('x')
plt.ylabel('bessel_j0(20x)')
plt.grid(True)


plt.grid(True)
plt.show()