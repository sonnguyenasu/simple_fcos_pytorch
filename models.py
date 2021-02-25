import torch
from torch import nn 
from torchvision.models import resnet50

class Backbone(nn.Module):
  def __init__(self,fpn=True):
    super(Backbone, self).__init__()
    self.fpn = fpn
    
    self.backbone = resnet50(pretrained=True,progress=True)
    self.layer0 = nn.Sequential(*list(self.backbone.children())[:4])
    self.layer1 = self.backbone.layer1
    self.layer2 = self.backbone.layer2
    self.layer3 = self.backbone.layer3
    self.layer4 = self.backbone.layer4
  def forward(self,x):
    if self.fpn:
      c1 = self.layer0(x)
      c2 = self.layer1(c1)
      c3 = self.layer2(c2)
      c4 = self.layer3(c3)
      c5 = self.layer4(c4)
      return c1,c2,c3,c4,c5
    else:
      return self.backbone(x)

class FPN(nn.Module):
  def __init__(self, start_at=512, fpn_dim=256):
    super(FPN,self).__init__()
    self.up1 = nn.Upsample(scale_factor=2)
    self.up2 = nn.Upsample(scale_factor=2)
    self.down1 = nn.Sequential(
        nn.Conv2d(fpn_dim,fpn_dim,1),
        nn.Conv2d(fpn_dim,fpn_dim,3,2,1),
        nn.Conv2d(fpn_dim,fpn_dim,1)
    )
    self.down2 = nn.Sequential(
        nn.Conv2d(fpn_dim,fpn_dim,1),
        nn.Conv2d(fpn_dim,fpn_dim,3,stride=2,padding=1),
        nn.Conv2d(fpn_dim,fpn_dim,1)
    )
    
    
    self.trans1= nn.Conv2d(start_at*4,fpn_dim,1)
    self.trans2= nn.Conv2d(start_at*2, fpn_dim,1)
    self.trans3= nn.Conv2d(start_at*1, fpn_dim,1)

    self.smooth1 = nn.Conv2d(256,256,1)
    self.smooth2 = nn.Conv2d(256,256,1)
  def forward(self,x):
    c3,c4,c5= x
    p5 = self.trans1(c5)
    p4 = self.trans2(c4) + self.up1(p5)
    p3 = self.trans3(c3) + self.up2(p4)
    p4 = self.smooth1(p4)
    p3 = self.smooth2(p3)
    p6 = self.down1(p5)
    p7 = self.down2(p6)
    return p3,p4,p5,p6,p7

class Head(nn.Module):
  def __init__(self,num_class):
    super(Head,self).__init__()
    self.num_class = num_class
    self.conv_cls = nn.Sequential(
        nn.Conv2d(256,256,1),
        nn.Conv2d(256,256,1),
        nn.Conv2d(256,256,1),
        nn.Conv2d(256,256,1),
    )
    self.conv_reg = nn.Sequential(
        nn.Conv2d(256,256,1),
        nn.Conv2d(256,256,1),
        nn.Conv2d(256,256,1),
        nn.Conv2d(256,256,1),
    )
    self.cls_branch = nn.Conv2d(256,num_class,1)
    self.center_branch = nn.Conv2d(256,1,1)
    self.reg_branch = nn.Conv2d(256,4,1)
  def forward(self,x):
    cls_center_f = self.conv_cls(x)
    reg_f = self.conv_reg(x)
    cls = self.cls_branch(cls_center_f)
    center = self.center_branch(cls_center_f)
    reg = torch.exp(self.reg_branch(reg_f))
    center = torch.nn.functional.sigmoid(center)
    cls = torch.nn.functional.sigmoid(cls)
    return torch.cat([cls,reg,center],dim=1)
