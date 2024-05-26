# Variational Autoencoder for MNIST

This project implements a Variational Autoencoder (VAE) to generate and reconstruct images from the MNIST dataset. The implementation includes data preprocessing, model definition, training, and testing.
[This blog](https://varunchitturi.com/notes/601290aa-3dab-47a9-b3dd-24c39467568a/) explains it further.

## Table of Contents

- [Installation](#Installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Generating New Images](#generating-new-images)
- [References](#references)

## Installation

Ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- torchvision
- matplotlib

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib
```

## Data Preparation

The MNIST dataset is downloaded and binarized. Binarization is performed by converting pixel values greater than 128 to 1 and others to 0.

```python
train.data = torch.where(train.data > 128, 1, 0)
test.data = torch.where(test.data > 128, 1, 0)
```

## Model Architecture

The VAE model consists of an encoder, a decoder, and a sampling function. 
The encoder compresses the input image into a latent space by first modeling the probability of 
the latent space and then sampling from it. 
The decoder then reconstructs the image from the latent 
space.

We chose to use a mix of non-linearities.  The most important part is that we have `tanh`
non-linearities at the end of our probabilistic encoder network layers (in order to constrain the outputs between -1 and 1) and a `sigmoid` non-linearity at the end of the decoder layer (to allow for good inputs for the `log` term in the loss function).

```python
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = 12
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(64, self.latent_dim ),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(64, self.latent_dim ),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim ,64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim,),
            nn.Sigmoid()
        )

    def get_latent_distribution(self, x):
        mu = self.mu(self.encoder(x))
        std = self.std(self.encoder(x))
        return mu, std

    def sample_latent_distribution(self, mu, std):
        return std * torch.randn(std.shape[0]).unsqueeze(-1).to(device) + mu

    def encode(self, x):
        mu, std = self.get_latent_distribution(x)
        return self.sample_latent_distribution(mu, std), mu, std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x_hat, mu, std = self.encode(x)
        return self.decode(x_hat), mu, std

    def generate(self):
        z = self.sample_latent_distribution(torch.zeros(self.latent_dim ).to(device), torch.ones
        (self.latent_dim).to(device))
        size = int(math.sqrt(self.input_dim))
        return self.decode(z).reshape(z.shape[0], size, size)

```

## Training

The model is trained using the Evidence Lower Bound (ELBO) loss, which includes a reconstruction 
term (Binary Cross Entropy Loss) and a regularization (latent space organization) term (KL 
Divergence).

```python

def elbo(x_hat, x, mu, std):
    bce_loss = nn.BCELoss(reduction='sum')
    kl_divergence = 0.5 * torch.sum(mu ** 2 + std ** 2 - torch.log(std ** 2))
    return bce_loss(x_hat, x) + 5 * kl_divergence

vae = VAE(28**2).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train.data, batch_size=batch_size, shuffle=True)

epochs = 15

for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, x in enumerate(train_loader):
        optimizer.zero_grad()
        x = torch.flatten(x.float(), start_dim=1).to(device)
        x_hat, mu, std = vae(x)
        loss = elbo(x_hat, x, mu, std)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Epoch: {}, Loss: {}'.format(epoch + 1, epoch_loss / len(train_loader)))

```

## Generating New Images

The VAE can generate new images by sampling from the latent space and passing the samples through the decoder.

```python
num_images = 16
num_columns = 4
f, axs = plt.subplots(4, num_images // num_columns)
axs = axs.flatten()
for i, ax in enumerate(axs):
    size = int(math.sqrt(vae.input_dim))
    ax.imshow(vae.generate().detach().cpu()[0], cmap='gray')
plt.show()
```

## References
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970)
- [Variational Autoencoder](https://arxiv.org/abs/1312.6114)
- [Understanding Variation Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [Variational Autoencoders Blog](https://mbernste.github.io/posts/vae/)
- [Variational-Autoencoder-for-MNIST](https://github.com/williamcfrancis/Variational-Autoencoder-for-MNIST)

This README provides an overview of the VAE project. For detailed implementation and experiments, refer to the project code.
