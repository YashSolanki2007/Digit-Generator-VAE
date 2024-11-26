'''
This is a simple implementation of a Variational Autoencoder from the paper 
'Auto-Encoding Variational Bayes' by Kingma et.al 
'''


'''
Notes from the paper

Notation
1. Phi: Encoder paramters
2. Theta: Decoder parameters
'''

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/sample_data/mnist_train_small.csv")

df.head()

len(df)

train_images = torch.stack(([torch.tensor(df.drop(columns=['6']).iloc[i]) for i in range(len(df))]))
train_images = train_images / 255.0
train_dataset = torch.utils.data.TensorDataset(train_images)
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=32)

train_images.shape

'''
z - Latent or the coding dimension variable
Encoder - q(z|x)
Decoder - p(x|z)


Structure of VAE

The outputs of the encoder: Mean and standard deviation
Use Epsilon, Mean and Std-dev to compute the conding dim
Use the coding dim and feed it into the decoder


'''

LATENT_DIM = 20

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.d1 = nn.Linear(784, 500)
    self.a1 = nn.ReLU()
    self.d2 = nn.Linear(500, 200)
    self.a2 = nn.ReLU()
    self.d3 = nn.Linear(200, 100)
    self.a3 = nn.ReLU()

    # Getting the mean
    self.d4 = nn.Linear(100, LATENT_DIM)

    # Getting the log variance
    self.d5 = nn.Linear(100, LATENT_DIM)

  def forward(self, x):
    x = self.a1(self.d1(x))
    x = self.a2(self.d2(x))
    h = self.a3(self.d3(x))
    mean = self.d4(h)
    log_var = self.d5(h)
    var = torch.exp(log_var / 2)

    # Apply reparemeterization here (z = mu + (eps * sigma))
    epsilon = torch.randn(mean.shape)
    z = mean + (epsilon * var)

    return z, mean, log_var

'''
Decoder information

Shapes of the output projections are going to be the opposite of the encoder
The output layer will have sigmoid activation function
'''

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.d1 = nn.Linear(LATENT_DIM, 100)
    self.a1 = nn.ReLU()
    self.d2 = nn.Linear(100, 200)
    self.a2 = nn.ReLU()
    self.d3 = nn.Linear(200, 500)
    self.a3 = nn.ReLU()
    self.d4 = nn.Linear(500, 784)
    self.a4 = nn.Sigmoid()

  def forward(self, x):
    x = self.a1(self.d1(x))
    x = self.a2(self.d2(x))
    x = self.a3(self.d3(x))
    return self.a4(self.d4(x))

class VAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    z, mean, log_var = self.encoder(x)
    return self.decoder(z), mean, log_var, z

model = VAE()
optim = torch.optim.Adam(model.parameters())

for i in train_dataset:
  print(i)
  print(len(i))
  print(i[0])
  print(type(i[0]))
  break

epochs = 500

for i in range(epochs):
  epoch_loss = 0
  for batch in train_dataset:
    batch = batch[0].to(torch.float32)
    dec_out, mean, log_var, z = model(train_images)
    var = torch.exp(log_var)
    kl_div = -0.5 * torch.sum(1 + log_var - (torch.pow(mean, 2)) - var)
    loss = kl_div + torch.nn.functional.binary_cross_entropy(dec_out, train_images, reduction='sum')
    epoch_loss += loss.item()
    optim.zero_grad()
    loss.backward()
    optim.step()
  print(f"Epoch: {i+1} - Loss: {epoch_loss}")

# Generating new numbers using the VAE
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    dec_out, _, _, _ = model(train_images)
    dec_out = dec_out.view(-1, 1, 28, 28)

plt.imshow(dec_out[0].permute(1, 2, 0), cmap='binary')

# Generating a unique image

model.eval()  # Set to evaluation mode
with torch.no_grad():
    # Sample a random latent vector
    unique_z = torch.randn(1, LATENT_DIM)
    unique_image = model.decoder(unique_z)
    unique_image = unique_image.view(28, 28).cpu().numpy()  # Reshape for visualization


plt.imshow(unique_image, cmap='binary')
plt.axis('off')
plt.show()

