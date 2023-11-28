from einops import rearrange, reduce, repeat
import torch
import pytorch_lightning as LIT
MAX_EPOCHS = 10

class Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size, stride, padding)
        
    def forward(self, x):
        return self.pool(x)


class Chiros(LIT.LightningModule):
    def __init__(self):
        super().__init__()
        self.channels = [3, 64, 128, 256, 512]
        
        layers = []

        for i in range(len(self.channels) - 1):
            layers.append(Layer(self.channels[i], self.channels[i+1]))
            layers.append(Layer(self.channels[i+1], self.channels[i+1]))
            layers.append(Downsample(self.channels[i+1], self.channels[i+1]))

        self.model = torch.nn.ModuleList(layers)

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )

        self.loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        x = reduce(x, 'b c h w -> b c', 'mean')
   
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid_loss'
        }
    
chiros = Chiros()