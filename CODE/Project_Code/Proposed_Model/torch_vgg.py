from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        self.slice_a = torch.nn.Sequential()
        self.slice_b = torch.nn.Sequential()
        self.slice_c = torch.nn.Sequential()
        
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])         
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
            
        for x in range(5):
            self.slice_a.add_module(str(x), vgg_pretrained_features[x])         
        for x in range(5, 10):
            self.slice_b.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.slice_c.add_module(str(x), vgg_pretrained_features[x])    
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        m = self.slice_a(X)
        m_1 = m
        m = self.slice_b(m)
        m_2 = m
        m = self.slice_c(m)
        m_3 = m
        
         
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        vgg_outputs1 = namedtuple("VggOutputs1", ['m_1', 'm_2', 'm_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        out1 = vgg_outputs1(m_1, m_2, m_3)
        return out, out1


if __name__=='__main__':
   vgg = Vgg16()
   print(vgg)
   """
   for i, (name, paramter) in enumerate(vgg.named_parameters()):
       print(i, name)
   """
