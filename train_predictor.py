# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import PearsonCorrCoef
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Pre-trained model')
# Add argument for relu with default value False
parser.add_argument('--relu', action='store_true', help='Use ReLU activation function')
parser.add_argument('--loss', help='Select the loss function to use', default="both")
args = parser.parse_args()

print(args)


# define your neural net here:

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = []

        self.layers.append(nn.Linear(self.input_size, 1024))
        if args.relu:
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(1024, 128))
        if args.relu:
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(128, 64))
        if args.relu:
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.1))
        self.layers.append(nn.Linear(64, 16))
        if args.relu:
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(16, 1))
        print(len(self.layers))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# load the training data

x = np.load("x_ModaDataset_CLIP_L14_embeddings.npy")
print(x.shape)

y = np.load("y_ratings_7.npy")

val_percentage = 0.05  # 5% of the trainingdata will be used for validation

train_border = int(x.shape[0] * (1 - val_percentage))

train_tensor_x = torch.Tensor(x[:train_border])  # transform to torch tensor
train_tensor_y = torch.Tensor(y[:train_border])

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)  # create your datset
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=1)  # create your dataloader

val_tensor_x = torch.Tensor(x[train_border:])  # transform to torch tensor
val_tensor_y = torch.Tensor(y[train_border:])

'''
print(train_tensor_x.size())
print(val_tensor_x.size())
print( val_tensor_x.dtype)
print( val_tensor_x[0].dtype)
'''

val_dataset = TensorDataset(val_tensor_x, val_tensor_y)  # create your datset
val_loader = DataLoader(val_dataset, batch_size=146, num_workers=1)  # create your dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP(x.shape[1]).to(device)  # CLIP embedding dim is 768 for CLIP ViT L 14

pearson = PearsonCorrCoef().to(device)

optimizer = torch.optim.Adam(model.parameters())

# choose the loss you want to optimze for
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

epochs = 50

model.train()
best_loss = 999

for epoch in tqdm(range(epochs)):
    losses = []
    losses2 = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        lossmse = criterion(output, y)
        lossmae = criterion2(output, y)
        if args.loss == "mae":
            lossmae.backward()
        elif args.loss == "mse":
            lossmse.backward()
        else:
            (lossmae + lossmse).backward()

        optimizer.step()

    # print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))
    losses = []
    losses2 = []
    target_total = []
    predicted_total = []

    with torch.no_grad():
        for batch_num, input_data in enumerate(val_loader):
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            lossMAE = criterion2(output, y)
            losses.append(loss.item())
            losses2.append(lossMAE.item())

            target_total.append(y.squeeze())
            predicted_total.append(output.squeeze())

    target_total = torch.stack(target_total, dim=0).flatten()
    predicted_total = torch.stack(predicted_total, dim=0).flatten()
    loss = criterion(predicted_total, target_total)
    lossMAE = criterion2(predicted_total, target_total)

    correlation = pearson(predicted_total, target_total)

    if sum(losses) / len(losses) < best_loss:
        # print("Best MAE Val loss so far. Saving model")
        best_mse = loss.item()
        best_mae = lossMAE.item()
        best_correlation = correlation
        best_loss = sum(losses) / len(losses)

print('MSE Loss %6.2f' % best_mse)
print('MAE Loss %6.2f' % best_mae)
print('Correlation %6.2f' % best_correlation)
print("_____________________")

