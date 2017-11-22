from PIL import Image
from skimage import color
from scipy.ndimage import zoom

import torch
import torch.utils.data as data
import torch.nn as nn

import numpy as np


class ColorizationDataset(data.Dataset):
    def __init__(self, img_names, preprocessor, image_size_in=256, image_size_out=64):
        super(ColorizationDataset, self).__init__()
        self.image_size_in = image_size_in
        self.image_size_out = image_size_out
        self.img_names = img_names
        self.preprocessor = preprocessor
    
    def __getitem__(self, index):
        image = Image.open(self.img_names[index])
        image = image.resize((self.image_size_in, self.image_size_in))
        image_rgb = np.array(image)
        image_lab = color.rgb2lab(image_rgb)
        image_lab = image_lab.transpose(2, 0, 1)
        
        input, label = image_lab[0, :, :] - 50, image_lab[1:, :, :]
        label = zoom(label, (1, self.image_size_out / self.image_size_in, self.image_size_out / self.image_size_in))
        
        label = self.preprocessor(label)
        return torch.FloatTensor(input).unsqueeze(0), torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.img_names)

class ImageColorizer1(nn.Module):
    def __init__(self, n_outputs, temperature=0.38):
        super(ImageColorizer1, self).__init__()
        self.n_outputs = n_outputs
        self.temperature = temperature
        
        self.moduleList = nn.ModuleList()
        self.moduleList.append(self.conv_block(1, 64, 2, downsample=True))
        self.moduleList.append(self.conv_block(64, 128, 2, downsample=True))
        self.moduleList.append(self.conv_block(128, 256, 3, downsample=True))
        self.moduleList.append(self.conv_block(256, 512, 3))
        self.moduleList.append(self.conv_block(512, 512, 3, dilation=2, padding=2))
        self.moduleList.append(self.conv_block(512, 512, 3, dilation=2, padding=2))
        self.moduleList.append(self.conv_block(512, 512, 3))
        self.moduleList.append(self.deconv_block())
                         
        self.conv_class = nn.Conv2d(256, n_outputs, 1)
        self.bn = nn.BatchNorm2d(n_outputs)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def conv_block(self, in_dim, out_dim, n_convs, dilation=1, padding=1, downsample=False):
        block = []
        for i in range(0, n_convs):
            if i == 0:
                in_feat, stride = in_dim, 1
            elif i == n_convs - 1:
                in_feat, stride = out_dim, 2 if downsample else 1
            else:
                in_feat, stride = out_dim, 1
                
            block.append(nn.Conv2d(in_feat, out_dim, 3, padding=padding, stride=stride, dilation=dilation))
            block.append(nn.ReLU(inplace=True))

        block.append(nn.BatchNorm2d(out_dim))
        return nn.Sequential(*block)
    
    def deconv_block(self):
        block = []
        block.append(nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.Conv2d(256, 256, 3, padding=1))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.Conv2d(256, 256, 3, padding=1))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.BatchNorm2d(256))
        return nn.Sequential(*block)
    
    def forward(self, input):
        for m in self.moduleList:
            input = m(input)

        input = self.conv_class(input)
        # input *= 1 / self.temperature  
        # input = self.bn(input)
        input = self.sigmoid(input)
        return input