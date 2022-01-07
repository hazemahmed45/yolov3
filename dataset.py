import io
from numpy.core.fromnumeric import ndim
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import numpy as np
import os
from PIL import Image,ImageFile
from utils import iou_width_height as iou , non_max_suppression as nms

ImageFile.LOAD_TRUNCATED_IMAGES=True

class VocDataset(Dataset):
    def __init__(self,csv_file,img_dir,label_dir,anchors,S=[13,26,52],C=20,transform=None):
        super().__init__()
        self.annotations=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.transform=transform
        self.C=C
        self.S=S
        self.anchors=torch.tensor(anchors[0]+anchors[1]+anchors[2])
        self.num_anchors=self.anchors.shape[0]
        self.num_anchors_per_scale=self.num_anchors//3
        self.ignore_iou_thresh=0.5
    def __getitem__(self, index) :
        label_path=os.path.join(self.label_dir,self.annotations.iloc[index,1])
        bboxes=np.roll(np.loadtxt(fname=label_path,delimiter=" ",ndmin=2),4,axis=1).tolist()
        img_path=os.path.join(self.img_dir,self.annotations.iloc[index,0])
        img=np.array(Image.open(img_path).convert('RGB'))
        if(self.transform is not None):
            augmented_dict=self.transform(image=img,bboxes=bboxes)
            img=augmented_dict['image']
            bboxes=augmented_dict['bboxes']
        targets = [torch.zeros((self.num_anchors//3,S,S,6)) for S in self.S] # [num of anchors,size,size,bbbox] -> bbox=[object prob,x,y,w,h,class]
        for box in bboxes:
            iou_anchors= iou(torch.tensor(box[2:4]),self.anchors)
            anchor_indices=iou_anchors.argsort(descending=True,dim=0)
            x,y,w,h,class_label=box
            has_anchor=[False,False,False]
            for anchor_idx in anchor_indices:
                scale_idx=anchor_idx//self.num_anchors_per_scale
                anchor_on_scale=anchor_idx%self.num_anchors_per_scale
                S=self.S[scale_idx]
                i,j=int(S*y),int(S*x)
                anchor_taken=targets[scale_idx][anchor_on_scale,i,j,0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale,i,j,0]=1
                    x_cel,y_cell=S*x-j,S*y-y
                    w_cell,h_cell=w*S,h*S
                    box_coords=torch.tensor([x_cel,y_cell,w_cell,h_cell])
                    targets[scale_idx][anchor_on_scale,i,j,1:5]=box_coords
                    targets[scale_idx][anchor_on_scale,i,j,5]=int(class_label)
                elif(not anchor_taken and iou_anchors[anchor_idx]>self.ignore_iou_thresh):
                    targets[scale_idx][anchor_on_scale,i,j,0]=-1
        return img, tuple(targets)
    def __len__(self):
        return len(self.annotations)