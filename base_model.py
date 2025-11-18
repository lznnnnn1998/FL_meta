import torch
import torch.nn as nn




class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.encoder = nn.Sequential(self.conv1, self.conv2)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, output_padding=1, padding=1)

        self.decoder = nn.Sequential(self.deconv1, self.deconv2)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x_hat = self.decoder(self.encoder(x))
        return x_hat