'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import sys
import torch.nn as nn
import torch.nn.init as init
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg11_client', 'vgg11_server', 'vgg9_client', 'vgg9_server'
]



class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x





class VGG_client(nn.Module):
    '''
    VGG Client-side model 
    '''
    def __init__(self, features):
        super(VGG_client, self).__init__()
        self.features = features
        
        # self.classifier = nn.Sequential()
        #  Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


class VGG_server(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG_server, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.zero_()
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_server_fn(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG_server_fn, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.zero_()
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_server_simple(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG_server_simple, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 128,),
            nn.ReLU(True),
            nn.Linear(128, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNet_client(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_client, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.features(x)
    #     return x

class AlexNet_server(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet_server, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG_client_test(nn.Module):
    '''
    VGG Client-side model 
    '''
    def __init__(self, features):
        super(VGG_client_test, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
        #     nn.Linear(512, 10),
        )
        # self.classifier = nn.Sequential()
        #  Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x




class VGG_server_test(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG_server_test, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.zero_()
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v,track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_client(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v,track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_server(cfg, batch_norm=False):
    layers = []
    in_channels = 64
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v,track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_client_fn(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v,track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_server_fn(cfg, batch_norm=False):
    layers = []
    in_channels = 64
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v,track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_server_simple(cfg, batch_norm=False):
    layers = []
    in_channels = 32
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_sf_client = {
    'A': [64, 'M'],
    'B': [64, 64, 'M'],
    'C': [64, 64, 'M'],
    'D': [64, 64, 'M'],
    'E': [64, 64, 'M'],
    'F': [64, 64, 'M'],
    'G': [64, 64, 'M'],
    'H': [64, 64, 'M'],
    'I': [64, 'M'],
}

cfg_sf_server = {
    'A': [ 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'A': [ 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'B': [ 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [ 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [ 128, 128, 'M', 256,256, 256, 256, 'M', 512,512,512, 512, 'M', 512,512,512, 512, 'M'],
    'E': [ 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
    'F': [ 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512],
    'G':[ 128, 128, 'M', 256,256, 256, 256, 'M', 512,512,512, 512, 'M', 512,512,512, 512],
    'H': [ 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'I': [ 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
}







cfg_sf_client_vgg9 = {
    'A': [32, 'M'],
}

cfg_sf_server_vgg9 = {
    'A': [ 32, 'M', 64, 64, 'M', 128, 128, 'M'],
}



def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

#VGG 11-layer client-side model 
def vgg11_client():
    return VGG_client(make_layers_client(cfg_sf_client['A'], batch_norm=True))

def vgg11_server():
    return VGG_server(make_layers_server(cfg_sf_server['A'], batch_norm=True))

def vgg19_client():
    return VGG_client(make_layers_client(cfg_sf_client['E'], batch_norm=True))

def vgg19_server():
    return VGG_server(make_layers_server(cfg_sf_server['E'], batch_norm=True))

def vgg16_client_fn():
    return VGG_client(make_layers_client_fn(cfg_sf_client['G'], batch_norm=True))
def vgg16_server_fn():
    return VGG_server_fn(make_layers_server_fn(cfg_sf_server['G'], batch_norm=True))

def vgg13_client_fn():
    return VGG_client(make_layers_client_fn(cfg_sf_client['H'], batch_norm=True))
def vgg13_server_fn():
    return VGG_server_fn(make_layers_server_fn(cfg_sf_server['H'], batch_norm=True))

def vgg11_client_fn():
    return VGG_client(make_layers_client_fn(cfg_sf_client['I'], batch_norm=True))
def vgg11_server_fn():
    return VGG_server_fn(make_layers_server_fn(cfg_sf_server['I'], batch_norm=True))


def vgg19_client_fn():
    return VGG_client(make_layers_client_fn(cfg_sf_client['F'], batch_norm=True))
def vgg19_server_fn():
    return VGG_server_fn(make_layers_server_fn(cfg_sf_server['F'], batch_norm=True))

def vgg16_client():
    return VGG_client(make_layers_client(cfg_sf_client['C'], batch_norm=True))

def vgg16_server():
    return VGG_server(make_layers_server(cfg_sf_server['C'], batch_norm=True))




def vgg9_client():
    return VGG_client(make_layers_client(cfg_sf_client_vgg9['A'], batch_norm=True))

def vgg9_server():
    return VGG_server_simple(make_layers_server_simple(cfg_sf_server_vgg9['A'], batch_norm=True))

def alex_client():
    return VGG_client(make_layers_client(cfg_sf_client['A']))

def alex_server():
    return AlexNet_server(num_classes=10)

def alex():
    return -1