import torch




def intersection_over_union(bboxes_preds,bboxes_labels,box_format='midpoint'):
    if(box_format=='midpoint'):
        pred_x1=bboxes_preds[..., 0:1]-bboxes_preds[..., 2:3]/2
        pred_y1=bboxes_preds[...,1:2]-bboxes_preds[..., 3:4]/2
        pred_x2=bboxes_preds[...,0:1]+bboxes_preds[..., 2:3]/2
        pred_y2=bboxes_preds[...,1:2]+bboxes_preds[..., 3:4]/2
        
        label_x1=bboxes_labels[..., 0:1]-bboxes_labels[..., 2:3]/2
        label_y1=bboxes_labels[...,1:2]-bboxes_labels[..., 3:4]/2
        label_x2=bboxes_labels[...,0:1]+bboxes_labels[..., 2:3]/2
        label_y2=bboxes_labels[...,1:2]+bboxes_labels[..., 3:4]/2
    if(box_format=='corners'):
        pred_x1=bboxes_preds[..., 0:1]
        pred_y1=bboxes_preds[...,1:2]
        pred_x2=bboxes_preds[...,2:3]
        pred_y2=bboxes_preds[...,3:4]
        
        label_x1=bboxes_labels[..., 0:1]
        label_y1=bboxes_labels[...,1:2]
        label_x2=bboxes_labels[...,2:3]
        label_y2=bboxes_labels[...,3:4]
    
    x1=torch.max(pred_x1,label_x1)
    x2=torch.min(pred_x2,label_x2)
    y1=torch.max(pred_y1,label_y1)
    y2=torch.min(pred_y2,label_y2)
    
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    pred_area=abs((pred_x2-pred_x1)*(pred_y2-pred_y1))
    label_area=abs((label_x2-label_x1)*(label_y2-label_y1))
    
    
    return intersection/(pred_area+label_area-intersection+1e-6)