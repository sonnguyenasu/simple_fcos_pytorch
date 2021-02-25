from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np
import math
import torch
class Base(Dataset):
    def __init__(self, image_path, label_path, num_class):
        self.image_path = image_path
        self.label_path = label_path
        self.images = [os.path.join(self.image_path, img) \
                        for img in os.listdir(self.image_path)]
        self.eps = 1e-8
        self.num_classes = num_class

    def _coordinate_map(self, img_height, img_width):
        x = np.arange(img_width)
        y = np.arange(img_height)
        x, y = np.meshgrid(x, y)
        x_lin = x.reshape((1, x.size))
        y_lin = y.reshape((1, x.size))
        coord_mat = np.concatenate((x_lin, y_lin), 0)
        return coord_mat

    def _get_vertices(self,bboxes):
        vertices = []
        for box in bboxes:
            vertices.append([box[0],box[1],box[0]+box[2],box[1]+box[3]])
        return vertices

    def _convert_vertices_to_label(self, clses, vertices,img_height,img_width):
        geo_map = np.zeros((img_height,img_width,4),dtype=np.float32)
        center_map = np.zeros((img_height,img_width),dtype=np.float32)
        class_map = np.zeros((img_height,img_width,self.num_classes))
        all_boxes = np.zeros((img_height, img_width))
        #coord = self._coordinate_map(img_height, img_width)
        x = np.arange(img_width)
        y = np.arange(img_height)
        index_x, index_y = np.meshgrid(x, y)
        
        for idx, vertice in enumerate(vertices): #vertice of shape (x0,y0,x1,y1)
            temp_mask = np.zeros((img_height, img_width)) #array to store all point in given box
            poly = np.array([[vertice[0],vertice[1]],[vertice[2],vertice[1]],[vertice[2],vertice[3]],[vertice[0],vertice[3]]])
            cv2.fillPoly(temp_mask, [poly],1)
            
            temp_mask = np.where(temp_mask>all_boxes,1,0)
            class_map[:,:,clses[idx]-1] += temp_mask.copy()*(clses[idx])
            d1 = index_x-vertice[0]
            d1[d1<0] = 0
            d2 = index_y-vertice[1]
            d2[d2<0] = 0
            d3 = vertice[2]-index_x
            d3[d3<0] = 0
            d4 = vertice[3]-index_y
            d4[d4<0] = 0
            geo_map[:,:,0] += d1*temp_mask
            geo_map[:,:,1] += d2*temp_mask
            geo_map[:,:,2] += d3*temp_mask
            geo_map[:,:,3] += d4*temp_mask
            min_lr = np.where(d1>d3,d3,d1)
            #print(min_lr.max())
            inv_max_lr = np.where(d1>d3, 1/(d1+self.eps), 1/(d3+self.eps))
            min_tb = np.where(d2>d4,d4,d2)
            inv_max_tb = np.where(d2>d4,1/(d2+self.eps),1/(d4+self.eps))
            center_map += np.sqrt(min_lr*inv_max_lr*min_tb*inv_max_tb)*temp_mask
            cv2.fillPoly(all_boxes, [poly],1)
        #return geo_map, center_map
        #print(class_map.shape)
        class_map = np.sum(class_map,axis=2)
        
        #class_map = np.expand_dims(class_map,axis=2)
        return torch.Tensor(class_map), torch.Tensor(geo_map).permute(2,0,1),\
             torch.Tensor(center_map)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img = cv2.imread(self.images[idx])
        return img

class FirstDataset(Base):
    def __init__(self, image_path, label_path, scales=(8,16,32,64,128), num_class=80):
        super(FirstDataset,self).__init__(image_path, label_path, num_class)
        self.labels = [os.path.join(self.label_path,image[:-3]+'txt') \
                        for image in os.listdir(self.image_path)]
        self.scales =scales
    def __getitem__(self,idx):
        img = cv2.imread(self.images[idx])
        if img is None:
          print(self.images[idx])  
        img = cv2.resize(img,(1024,800))
        labels = open(self.labels[idx],'r').readlines()
        bboxes = []
        for label in labels:
            dat = [float(num) for num in label.split(' ')]
            bboxes.append(dat)
        # sort the box in their area order
        bboxes.sort(key=lambda x: x[3]*x[4])
        clses = [int(box[0]+1) for box in bboxes]
        #print(clses)
        bboxes = [[int(box[1]*1024-box[3]/2*1024),int((box[2]-box[4]/2)*800)\
            ,int(box[3]*1024),int(box[4]*800)] for box in bboxes]
        H, W = img.shape[:2]
        geo_map= []
        center_map = []
        class_map = []
        #getting the vertices of the bounding boxes
        for scale in self.scales:
            boxes = np.array(bboxes) / scale
            boxes = boxes.astype('int')
            vertices = self._get_vertices(boxes.tolist())
            cl_map, g_map, c_map = self._convert_vertices_to_label(clses,\
                                            vertices,math.ceil(H/scale), \
                                            math.ceil(W/scale))
            geo_map.append(g_map)
            center_map.append(c_map)
            class_map.append(cl_map)
        return torch.Tensor(img/255).permute(2,0,1), class_map, geo_map, center_map
