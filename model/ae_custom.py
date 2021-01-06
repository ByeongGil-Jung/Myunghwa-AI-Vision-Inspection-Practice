from torch import nn

from model.base import ModelBase


class AutoencoderModelCustom01(ModelBase):

    def __init__(self):
        super(AutoencoderModelCustom01, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=32, stride=16, padding=8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=32, stride=16, padding=8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=32, stride=16, padding=8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=32, stride=16, padding=8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.init_weight()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
