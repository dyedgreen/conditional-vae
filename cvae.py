# Conditional variational auto-encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import os


# Helper functions

def idx_to_onehot(idx):
    one_hot = torch.zeros(idx.size(0), 10)
    for i in range(idx.size(0)):
        one_hot[i][idx[i]] = 1
    return one_hot


# Define our Model

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # We model as p(z|y, x); z ~ p(z|x) = p(z); p(y|z, x),
        # where x is the number and y the image. We could have
        # conditioned p(z) on x as well, but that requires
        # additional changes to the KL divergence part of
        # our loss function.

        self.encoder = nn.Sequential(
            nn.Linear(28**2 + 10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.mu = nn.Linear(512, 256)
        self.log_var = nn.Linear(512, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256 + 10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 28**2),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Find parameters for latent distribution
        x = idx_to_onehot(x)
        h = self.encoder(torch.cat((y.view(-1, 28**2), x), dim=1))
        mu = self.mu(h)
        z = mu

        if self.training:
            # Re-parametrization trick
            # (move sampling to input)
            log_var = self.log_var(h)
            eps = torch.randn_like(mu)
            z = eps.mul(log_var.mul(0.5).exp()).add_(mu)

        r = self.decoder(torch.cat((z, x), dim=1)).view(-1, 1, 28, 28)

        # Only return p(z|x,y) parameters if training
        if self.training:
            return (r, mu, log_var)
        return r

    def sample(self, x):
        # Sample one digit
        z = torch.randn(1, 256)
        one_hot = torch.zeros(1, 10)
        one_hot[0][x] = 1
        return self.decoder(torch.cat((z, one_hot), dim=1)).view(1, 28, 28)


# Define our loss function

def loss_fn(x, r, mu, log_var):
    # Reconstruction loss
    loss_r = F.binary_cross_entropy(r, x, reduction="sum")
    # KL Divergence, see appendix B
    loss_kl = - 0.5 * (1 + log_var - mu**2 - log_var.exp()).sum()
    return loss_r + loss_kl


# Fetch dataset

dataset = torchvision.datasets.MNIST(root="data-mnist", download=True, transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.MNIST(root="data-mnist", train=False, transform=torchvision.transforms.ToTensor())

loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1)


# Train for a bunch of epochs

model = Model()
adam = torch.optim.Adam(model.parameters(), lr=1e-3)

os.makedirs("results-cvae", exist_ok=True)

model.train()

for epoch in range(1, 11):
    loss_train = 0
    loss_test = 0

    for batch_i, (y, x) in enumerate(loader):
        adam.zero_grad()

        r, mu, log_var = model(x, y)

        l = loss_fn(y, r, mu, log_var)
        loss_train += l.item()
        l.backward()

        adam.step()

        print("Training Epoch {}: Batch [{}/{}] Loss: {}".format(
            epoch,
            batch_i + 1,
            len(loader),
            l.item() / y.size(0)
        ))

    for _, (y, x) in enumerate(loader_test):
        r, mu, log_var = model(x, y)
        l = loss_fn(y, r, mu, log_var)
        loss_test += l.item()

    print("-------> Epoch {}: Average Loss: {}".format(
        epoch,
        loss_train / len(dataset)
    ))
    print("-------> Epoch {}: Average Validation Loss: {}".format(
        epoch,
        loss_test / len(dataset_test)
    ))

    # Generate some nice output image
    torchvision.utils.save_image([model.sample(i%10) for i in range(10**2)], "results-cvae/sample_{:02d}.png".format(epoch), nrow=10)
