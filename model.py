import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16

import segmentation_models_pytorch as smp


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s,self).__init__()
        self.pretrained_model = vgg16(pretrained = True)
        features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])
        
        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256, num_classes, 1)
        
        # Score pool4        
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)        
        
        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )
        
        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        # UpScore2_pool4 using deconv
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, 
                                                 num_classes, 
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)
        
        # UpScore8 using deconv
        self.upscore8 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)
    
    def forward(self, x):
        pool3 = h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)
        
        h = self.conv(h)
        h = self.score_fr(h)
       
        score_pool3c = self.score_pool3_fr(pool3)    
        score_pool4c = self.score_pool4_fr(pool4)
        
        # Up Score I
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)
        
        # Sum II
        h = upscore2_pool4c + score_pool3c
        
        # Up Score III
        upscore8 = self.upscore8(h)
        
        return upscore8


def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class DeconvNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DeconvNet, self).__init__()
        
        # 224 x 224 conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # 112 x 112 conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # 56 x 56 conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # 28 x 28 conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # 14 x 14 conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # 7 x 7 fc6
        self.fc6 = CBR(512, 4096, 7, 1, 0)
        self.drop6 = nn.Dropout2d(0.5)

        # 1 x 1 fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d(0.5)

        # 7 x 7 fc6-deconv
        self.fc6_deconv = DCB(4096, 512, 7, 1, 0)

        # 14 x 14 unpool5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5_1 = DCB(512, 512, 3, 1, 1)
        self.deconv5_2 = DCB(512, 512, 3, 1, 1)
        self.deconv5_3 = DCB(512, 512, 3, 1, 1)

        # 28 x 28 unpool4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4_1 = DCB(512, 512, 3, 1, 1)
        self.deconv4_2 = DCB(512, 512, 3, 1, 1)
        self.deconv4_3 = DCB(512, 256, 3, 1, 1)

        # 56 x 56 unpool3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3_1 = DCB(256, 256, 3, 1, 1)
        self.deconv3_2 = DCB(256, 256, 3, 1, 1)
        self.deconv3_3 = DCB(256, 128, 3, 1, 1)

        # 112 x 112 unpool2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2_1 = DCB(128, 128, 3, 1, 1)
        self.deconv2_2 = DCB(128, 64, 3, 1, 1)

        # 224 x 224 unpool1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = DCB(64, 64, 3, 1, 1)
        self.deconv1_2 = DCB(64, 64, 3, 1, 1)

        # score
        self.score_fr = nn.Conv2d(64, num_classes, 1, 1, 0, 1)


    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h, pool1_indices = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h, pool2_indices = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h, pool3_indices = self.pool3(h)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h, pool4_indices = self.pool4(h)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h, pool5_indices = self.pool5(h)

        h = self.fc6(h)
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.drop7(h)

        h = self.fc6_deconv(h)

        h = self.unpool5(h, pool5_indices)
        h = self.deconv5_1(h)
        h = self.deconv5_2(h)
        h = self.deconv5_3(h)

        h = self.unpool4(h, pool4_indices)
        h = self.deconv4_1(h)
        h = self.deconv4_2(h)
        h = self.deconv4_3(h)

        h = self.unpool3(h, pool3_indices)
        h = self.deconv3_1(h)
        h = self.deconv3_2(h)
        h = self.deconv3_3(h)

        h = self.unpool2(h, pool2_indices)
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)

        h = self.unpool1(h, pool1_indices)
        h = self.deconv1_1(h)
        h = self.deconv1_2(h)
        
        h = self.score_fr(h)
        
        return h


class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True):
        super(SegNet, self).__init__()
        
        # conv1
        self.cbr1_1 = CBR(3, 64, 3, 1, 1)
        self.cbr1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv2
        self.cbr2_1 = CBR(64, 128, 3, 1, 1)
        self.cbr2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv3
        self.cbr3_1 = CBR(128, 256, 3, 1, 1)
        self.cbr3_2 = CBR(256, 256, 3, 1, 1)
        self.cbr3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv4
        self.cbr4_1 = CBR(256, 512, 3, 1, 1)
        self.cbr4_2 = CBR(512, 512, 3, 1, 1)
        self.cbr4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv5
        self.cbr5_1 = CBR(512, 512, 3, 1, 1)
        self.cbr5_2 = CBR(512, 512, 3, 1, 1)
        self.cbr5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # deconv5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr5_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_1 = CBR(512, 512, 3, 1, 1)

        # deconv4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr4_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_1 = CBR(512, 256, 3, 1, 1)

        # deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr3_3 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_2 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_1 = CBR(256, 128, 3, 1, 1)

        # deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr2_2 = CBR(128, 128, 3, 1, 1)
        self.dcbr2_1 = CBR(128, 64, 3, 1, 1)

        # deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr1_1 = CBR(64, 64, 3, 1, 1)

        # score
        self.score_fr = nn.Conv2d(64, num_classes, 3, 1, 1, 1)

    def forward(self, x):
        h = self.cbr1_1(x)
        h = self.cbr1_2(h)
        h, pool1_indices = self.pool1(h)

        h = self.cbr2_1(h)
        h = self.cbr2_2(h)
        h, pool2_indices = self.pool2(h)

        h = self.cbr3_1(h)
        h = self.cbr3_2(h)
        h = self.cbr3_3(h)
        h, pool3_indices = self.pool3(h)

        h = self.cbr4_1(h)
        h = self.cbr4_2(h)
        h = self.cbr4_3(h)
        h, pool4_indices = self.pool4(h)

        h = self.cbr5_1(h)
        h = self.cbr5_2(h)
        h = self.cbr5_3(h)
        h, pool5_indices = self.pool5(h)

        h = self.unpool5(h, pool5_indices)
        h = self.dcbr5_3(h)
        h = self.dcbr5_2(h)
        h = self.dcbr5_1(h)

        h = self.unpool4(h, pool4_indices)
        h = self.dcbr4_3(h)
        h = self.dcbr4_2(h)
        h = self.dcbr4_1(h)

        h = self.unpool3(h, pool3_indices)
        h = self.dcbr3_3(h)
        h = self.dcbr3_2(h)
        h = self.dcbr3_1(h)

        h = self.unpool2(h, pool2_indices)
        h = self.dcbr2_2(h)
        h = self.dcbr2_1(h)

        h = self.unpool1(h, pool1_indices)
        h = self.dcbr1_1(h)
        
        h = self.score_fr(h)
        
        return h


def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                        out_ch, 
                                        kernel_size=size, 
                                        stride=1, 
                                        padding=rate, 
                                        dilation=rate),
                             nn.ReLU())
    return conv_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(conv_relu(3, 64, 3, 1),
                                      conv_relu(64, 64, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features2 = nn.Sequential(conv_relu(64, 128, 3, 1),
                                      conv_relu(128, 128, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features3 = nn.Sequential(conv_relu(128, 256, 3, 1),
                                      conv_relu(256, 256, 3, 1),
                                      conv_relu(256, 256, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features4 = nn.Sequential(conv_relu(256, 512, 3, 1),
                                      conv_relu(512, 512, 3, 1),
                                      conv_relu(512, 512, 3, 1),
                                      nn.MaxPool2d(3, stride=1, padding=1))
                                      # and replace subsequent conv layer r=2
        self.features5 = nn.Sequential(conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1), 
                                      nn.AvgPool2d(3, stride=1, padding=1)) # 마지막 stride=1로 해서 두 layer 크기 유지 

    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        return out

    
class classifier(nn.Module):
    def __init__(self, num_classes): 
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(conv_relu(512, 1024, 3, rate=12), 
                                       nn.Dropout2d(0.5), 
                                       conv_relu(1024, 1024, 1, 1), 
                                       nn.Dropout2d(0.5), 
                                       nn.Conv2d(1024, num_classes, 1)
                                       )

    def forward(self, x):
        out = self.classifier(x)
        return out 


class DeepLabV1(nn.Module):
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV1, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.upsampling = upsampling

    def forward(self, x):
        x = self.backbone(x)
        _, _, feature_map_h, feature_map_w = x.size()
        x = self.classifier(x)
        x = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode='bilinear')
        
        return x


class Deeplab_V3_Resnet101(nn.Module):
    def __init__(self, num_classes):
        super(Deeplab_V3_Resnet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)

        self.model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        return self.model(x)['out']


class deeplabv3_resnet50(nn.Module):
    def __init__(self, num_classes):
        super(deeplabv3_resnet50, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        self.model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        return self.model(x)['out']


class Unet_resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Unet_resnet50, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)


class DeepLabV3Plus_resnet101(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus_resnet101, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)


class DeepLabV3Plus_efficientnet(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus_efficientnet, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)


class FPN_efficientnet(nn.Module):
    def __init__(self, num_classes):
        super(FPN_efficientnet, self).__init__()
        self.model = smp.FPN(
            encoder_name="efficientnet-b7",
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        return self.model(x)