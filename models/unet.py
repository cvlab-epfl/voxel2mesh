from itertools import chain
import torch.nn as nn 
import torch
import torch.nn.functional as F 
from IPython import embed
from utils.utils_common import crop_and_merge
from utils.utils_unet import UNetLayer
 


class UNet(nn.Module):
    """ The U-Net. """
    def __init__(self, config):

        super(UNet, self).__init__()
        self.config = config 

        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2)

        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.ConvTranspose2d

        '''  Down layers '''
        down_layers = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            lyr = UNetLayer(config.first_layer_channels * 2**(i - 1), config.first_layer_channels * 2**i, config.ndims, config.batch_norm)
            down_layers.append(lyr)

        ''' Up layers '''
        up_layers = []
        for i in range(config.steps - 1, -1, -1):
            upconv = ConvTransposeLayer(in_channels=config.first_layer_channels   * 2**(i+1), out_channels=config.first_layer_channels * 2**i, kernel_size=2, stride=2)
            lyr = UNetLayer(config.first_layer_channels * 2**(i + 1), config.first_layer_channels * 2**i, config.ndims, config.batch_norm)
            up_layers.append((upconv, lyr))

        ''' Final layer '''
        final_layer = ConvLayer(in_channels=config.first_layer_channels, out_channels=config.num_classes, kernel_size=1)

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.final_layer = final_layer

    def forward(self, data):

        x = data['x'].cuda()

        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)

        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)

        pred = self.final_layer(x)

        return pred


    def loss(self, data, epoch):

        pred = self.forward(data)
  
        CE_Loss = nn.CrossEntropyLoss()
        loss = CE_Loss(pred, data['y_voxels'].cuda())

        log = {"loss": loss.detach()}

        return loss, log
