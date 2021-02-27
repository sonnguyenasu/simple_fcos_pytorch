import torch
from fcos import FCOS 
import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
num_class = 2
url = 'https://cdn.mos.cms.futurecdn.net/hbKifQWBTcdhTEw8zsJWnF-1200-80.jpg'

resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.resize(image, (1024,800))
model = FCOS(num_class).to('cuda')
model.load_state_dict(torch.load('/content/model_inter.pth'))

image = torch.Tensor(image).permute(2,0,1).unsqueeze(0)
prediction = model(image.to('cuda'))[0]
cls_pred,reg_pred,center_pred = torch.split(prediction,[num_class,4,1],dim=1)
cls_pred = cls_pred[0,:,:,:].permute(1,2,0).detach().cpu().numpy()
reg_pred = reg_pred[0,:,:,:].permute(1,2,0).detach().cpu().numpy()
center_pred = center_pred[0,:,:,:].permute(1,2,0).detach().cpu().numpy()

plt.imshow(cls_pred[:,:,1],cmap='gray')
plt.colorbar()
plt.savefig('log.jpg')
