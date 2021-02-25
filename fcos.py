from models import *

class FCOS(nn.Module):
  def __init__(self,num_class):
    super(FCOS,self).__init__()
    self.backbone = Backbone()
    self.fpn = FPN()
    self.head = Head(num_class)
  def forward(self,x):
    features = self.backbone(x)
    levels = self.fpn(features[2:])
    res = []
    for level in levels:
      res.append(self.head(level))
    return res