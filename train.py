from fcos import FCOS 
from loss import Loss 
from dataset import FirstDataset
from torch.utils.data import DataLoader
import torch
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
base = FirstDataset('/content/export/images','/content/export/labels')
loader = DataLoader(base,batch_size=4,shuffle=True)
num_class = 1
EPOCH = 10
criterion = Loss(num_class=num_class).to(device)
model = FCOS(num_class).to(device)
optimizer = torch.optim.Adam(model.parameters(),1e-4)

for e in range(EPOCH):
  total_loss,cls_loss,reg_loss,center_loss = 0,0,0,0
  for idx, (img, class_map_, geo_map_, center_map_) in enumerate(loader):
    img  = img.to(device)
    total_loss_,cls_loss_,reg_loss_,center_loss_ = 0,0,0,0
    predictions = model(img)
    for j in range(5):
      class_map, geo_map, center_map = class_map_[j].to(device), geo_map_[j].to(device), center_map_[j].to(device)
      prediction = predictions[j]
      #print(geo_map.shape)
      target = [class_map,geo_map,center_map]
      losses = criterion(prediction,target)
      total_loss_ += losses[0]
      cls_loss_ += losses[1]
      center_loss_ += losses[2]
      reg_loss_ += losses[3]
      #del class_map,geo_map,center_map,losses,prediction
      #torch.cuda.empty_cache()
    optimizer.zero_grad()
    total_loss_.backward()
    optimizer.step()
    total_loss += total_loss_
    cls_loss += cls_loss_
    center_loss += center_loss_
    reg_loss += reg_loss_
    #if idx % 10 == 9:
  print('-'*89)
  print("Epoch: {}, iter: {}".format(e+1,idx+1))
  print(f'total:{total_loss.item()/(idx+1)}')
  print(f'cls:{cls_loss.item()/(idx+1)}')
  print(f'regression:{reg_loss.item()/(idx+1)}')
  print(f'center:{center_loss.item()/(idx+1)}')
  torch.save(model.state_dict(),'/content/model_inter.pth')
  #print(criterion(prediction,target))
#torch.cuda.empty_cache()