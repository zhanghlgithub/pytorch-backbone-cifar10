# -*- coding:utf-8 -*-
import torch.nn as nn
import sys
import math
sys.path.append(".")
import backbones
#print(backbones.__dict__)

class Models(nn.Module):
    
    def __init__(self, arch, embeding_size=128, num_classes=10):
        super(Models, self).__init__()
        
        self.model = backbones.__dict__[arch](embeding_size=embeding_size)

        self.fc1 = nn.Linear(embeding_size, num_classes)
        self._init_weights
        
    def forward(self, x):
        
        feature = self.feature(x)
        out = self.fc1(feature)
        return out
    
    def feature(self, x):
        feature = self.model.forward(x)
        return feature
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__=="__main__":
    
    print("start...")
    net = Models("mobileNetV2")
    print("end...")
    
    
    