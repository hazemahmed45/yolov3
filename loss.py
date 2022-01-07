import torch
from torch import nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse=nn.MSELoss()
        self.bce=nn.BCEWithLogitsLoss()
        self.entropy=nn.CrossEntropyLoss()
        self.sigmoid=nn.Sigmoid()
        
        #constants
        self.lambda_class=1
        self.lambda_noobj=10
        self.lambda_obj=1
        self.lambda_box=10
        
        return 
    def forward(self,predictions,targets,anchors):
        
        obj_mask=targets[...,0]==1
        noobj_mask=targets[...,0]==0
        
        #No Object Loss
        no_object_loss=self.bce(predictions[...,0:1][noobj_mask],targets[...,0:1][noobj_mask])
        
        #Object Loss
        anchors= anchors.reshape(1,3,1,1,2) 
        box_pred=torch.cat([self.sigmoid(predictions[...,1:3]),torch.exp(predictions[...,3:5])*anchors],dim=-1)
        ious=intersection_over_union(box_pred[obj_mask],targets[...,1:5][obj_mask]).detach()
        object_loss=self.bce(predictions[...,0:1][obj_mask],ious*targets[...,0:1][obj_mask])
        
        #Box Coordinates Loss
        predictions[...,1:3]=self.sigmoid(predictions[...,1:3]) # x,y between [0,1]
        targets[...,3:5]=torch.log(targets[...,3:5]/anchors + 1e-16)
        box_loss=self.mse(predictions[...,1:5][obj_mask],targets[...,1:5][obj_mask])
        
        #Class Loss
        class_loss=self.entropy(predictions[...,5:][obj_mask],targets[...,5][obj_mask].long())
        
        return self.lambda_box*box_loss+self.lambda_class*class_loss+self.lambda_noobj*no_object_loss+self.lambda_obj*object_loss