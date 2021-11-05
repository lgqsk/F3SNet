import torch.nn as nn
import torch
from models.parts import pooling, up, conv_bn_relu, ResBasicBlock
from models.parts import ResBasicBlock as BaseModule

class FFE_top_down(nn.Module):
    # Full-scale Feature Extractor(top->down)
    def __init__(self, image_channels, channels):
        super(FFE_top_down, self).__init__()
        self.input_conv = conv_bn_relu(image_channels,channels[0],3,1)
        
        self.conv1 = BaseModule(channels[0],channels[0])
        self.conv2 = BaseModule(channels[0],channels[1])
        self.conv3 = BaseModule(channels[0]*3,channels[2])
        self.conv4 = BaseModule(channels[0]*7,channels[3])
        self.conv5 = BaseModule(channels[0]*15,channels[4])
        
        self.smooth1 = conv_bn_relu(channels[1],channels[0],1,0)
        self.smooth2 = conv_bn_relu(channels[2],channels[1],1,0)
        self.smooth3 = conv_bn_relu(channels[3],channels[2],1,0)
        self.smooth4 = conv_bn_relu(channels[4],channels[3],1,0)
        self.smooth5 = conv_bn_relu(channels[5],channels[4],1,0)
        
    def forward(self, x, y):
        
        # Initial feature extraction
        x,y = self.input_conv(x),self.input_conv(y)
        
         # Feature Extraction Branch
        x1 = self.conv1(x)
        x2 = self.conv2(pooling(2)(x1))
        x3 = self.conv3(torch.cat([pooling(4)(x1), pooling(2)(x2)],1))
        x4 = self.conv4(torch.cat([pooling(8)(x1), pooling(4)(x2),pooling(2)(x3)],1))
        x5 = self.conv5(torch.cat([pooling(16)(x1),pooling(8)(x2),pooling(4)(x3),pooling(2)(x4)],1))
        
        y1 = self.conv1(y)
        y2 = self.conv2(pooling(2)(y1))
        y3 = self.conv3(torch.cat([pooling(4)(y1), pooling(2)(y2)],1))
        y4 = self.conv4(torch.cat([pooling(8)(y1), pooling(4)(y2),pooling(2)(y3)],1))
        y5 = self.conv5(torch.cat([pooling(16)(y1),pooling(8)(y2),pooling(4)(y3),pooling(2)(y4)],1))
        
        # Features of the same scale are concatenated
        x1 = self.smooth1(torch.cat([x1,y1],1))
        x2 = self.smooth2(torch.cat([x2,y2],1))
        x3 = self.smooth3(torch.cat([x3,y3],1))
        x4 = self.smooth4(torch.cat([x4,y4],1))
        x5 = self.smooth5(torch.cat([x5,y5],1))
        return x1,x2,x3,x4,x5

class FFE_bottom_up(nn.Module):
    # Full-scale Feature Extractor(bottom->up)
    def __init__(self, channels):
        super(FFE_bottom_up, self).__init__()
        self.up2_1 = up(channels[1],channels[0],2)
        self.up3_1 = up(channels[2],channels[0],4)
        self.up4_1 = up(channels[3],channels[0],8)
        self.up5_1 = up(channels[4],channels[0],16)
        self.up3_2 = up(channels[2],channels[1],2)
        self.up4_2 = up(channels[3],channels[1],4)
        self.up5_2 = up(channels[4],channels[1],8)
        self.up4_3 = up(channels[3],channels[2],2)
        self.up5_3 = up(channels[4],channels[2],4)
        self.up5_4 = up(channels[4],channels[3],2)
        
        self.Msmooth1 = conv_bn_relu(channels[0]*5,channels[0],1,0)
        self.Msmooth2 = conv_bn_relu(channels[1]*4,channels[1],1,0)
        self.Msmooth3 = conv_bn_relu(channels[2]*3,channels[2],1,0)
        self.Msmooth4 = conv_bn_relu(channels[3]*2,channels[3],1,0)
        self.Msmooth5 = conv_bn_relu(channels[4]*1,channels[4],1,0)
        
    def forward(self, x1, x2, x3, x4, x5):
        # The connections from bottom to up
        x1 = torch.cat([x1,self.up2_1(x2),self.up3_1(x3),self.up4_1(x4),self.up5_1(x5)],1)
        x2 = torch.cat([x2,self.up3_2(x3),self.up4_2(x4),self.up5_2(x5)],1)
        x3 = torch.cat([x3,self.up4_3(x4),self.up5_3(x5)],1)
        x4 = torch.cat([x4,self.up5_4(x5)],1)

        # Reduce the number of channels of feature maps
        x1 = self.Msmooth1(x1)
        x2 = self.Msmooth2(x2)
        x3 = self.Msmooth3(x3)
        x4 = self.Msmooth4(x4)
        x5 = self.Msmooth5(x5)
        
        return x1,x2,x3,x4,x5
    
class Decoder(nn.Module):
    # extracting difference information
    def __init__(self, channels, bilinear):
        super(Decoder, self).__init__()        
        self.conv5 = BaseModule(channels[4],channels[4])
        self.up5_4 = up(in_channels=channels[4], out_channels=channels[3], bilinear=bilinear, scale=2)
        self.conv4 = BaseModule(channels[4],channels[3])
        self.up4_3 = up(in_channels=channels[3], out_channels=channels[2], bilinear=bilinear, scale=2)
        self.conv3 = BaseModule(channels[3],channels[2])
        self.up3_2 = up(in_channels=channels[2], out_channels=channels[1], bilinear=bilinear, scale=2)
        self.conv2 = BaseModule(channels[2],channels[1])
        self.up2_1 = up(in_channels=channels[1], out_channels=channels[0], bilinear=bilinear, scale=2)
        self.conv1 = BaseModule(channels[1],channels[0])
        
    def forward(self, x1, x2, x3, x4, x5):
        x5 = self.conv5(x5)
        x4 = self.conv4(torch.cat([self.up5_4(x5),x4],1))
        x3 = self.conv3(torch.cat([self.up4_3(x4),x3],1))
        x2 = self.conv2(torch.cat([self.up3_2(x3),x2],1))
        x1 = self.conv1(torch.cat([self.up2_1(x2),x1],1))
        return x1,x2,x3,x4,x5
    
class FC(nn.Module):
    # Multi-scale Classifier
    def __init__(self,channels):
        super(FC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[0], 2, kernel_size=1, padding=0, bias=False),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[1], 2, kernel_size=1, padding=0, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[2], 2, kernel_size=1, padding=0, bias=False),
            nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[3], 2, kernel_size=1, padding=0, bias=False),
            nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[4], 2, kernel_size=1, padding=0, bias=False),
            nn.Upsample(scale_factor=16, mode='bilinear',align_corners=True),
        )
        
    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)
        
        return x1+x2+x3+x4+x5
    
class F3SNet(nn.Module):
    def __init__(self,image_channels,init_channels,bilinear):
        super(F3SNet, self).__init__()
        channels = [init_channels, init_channels*2, init_channels*4, init_channels*8, init_channels*16, init_channels*32]
        
        # Run these modules in turn
        self.FFE_top_down = FFE_top_down(image_channels,channels)
        self.FFE_bottom_up = FFE_bottom_up(channels)
        self.Decoder = Decoder(channels,bilinear)
        self.FC = FC(channels)
        
    def forward(self,x,y):
        x1,x2,x3,x4,x5 = self.FFE_top_down(x,y)
        x1,x2,x3,x4,x5 = self.FFE_bottom_up(x1,x2,x3,x4,x5)
        x1,x2,x3,x4,x5 = self.Decoder(x1,x2,x3,x4,x5)
        x1 = self.MC(x1,x2,x3,x4,x5)
        return x1
    