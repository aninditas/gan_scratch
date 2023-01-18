import torch.nn as nn
import torchvision


class Cls_model(nn.Module):

    def __init__(self, output_dim, arch='vgg', pretrained=False):
        super(Cls_model, self).__init__()
        self.n_classes = output_dim
        self.arch = arch
        self.classifier2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, output_dim))
        if arch=='vgg':
            self.net = torchvision.models.vgg16(pretrained=pretrained)# .vgg16(pretrained=pretrained)
        elif arch=='resnet':
            self.net=torchvision.models.resnet50(pretrained=pretrained)
        elif arch=='mobilenet':
            self.net=torchvision.models.mobilenet_v2(pretrained=pretrained)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        x1 = self.net(x)
        # print 'Passed Thru VGG'
        y = self.classifier2(x1)
        return y
