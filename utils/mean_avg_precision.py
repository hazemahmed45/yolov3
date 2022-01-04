import torch
from collections import Counter
import sys
sys.path.append('./utils')
from iou import intersection_over_union

def mean_average_precision(pred_bboxes,true_bboxes,iou_threshold,box_format='corners',num_classes=20):
    ## pred_bboxes: list = [[train_idx,class,conf,x1,y1,x2,y2],[],[]]
    average_precision=[]
    eps=1e-6
    
    for c in range(num_classes):
        detections=[bbox for bbox in pred_bboxes if bbox[1]==c]
        ground_truths=[bbox for bbox in true_bboxes if bbox[1]==c]
        
        amount_bboxes=Counter([gt[0] for gt in ground_truths])
        
        for key,val in amount_bboxes.items():
            amount_bboxes[key]=torch.zeros(val)
            
        detections.sort(key = lambda x : x[2],reverse=True)
        TP=torch.zeros(len(detections))
        FP=torch.zeros(len(detections))
        total_true_bboxes=len(ground_truths)
        
        for detection_idx,detection in enumerate(detections):
            ground_truth_img=[bbox for bbox in ground_truths if bbox[0] == detection[0]]
            
            num_gts=len(ground_truth_img)
            best_iou=0
            
            
            for gt_idx,gt_bbox in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),torch.tensor(gt_bbox[3:]),box_format=box_format)
                
                if(iou>best_iou):
                    best_iou=iou
                    best_gt_idx=gt_idx
            if(best_iou>iou_threshold):
                if(amount_bboxes[detection[0]][best_gt_idx]==0):
                    TP[detection_idx]=1
                    amount_bboxes[detection[0]][best_gt_idx]=1
                else:
                    FP[detection_idx]=1
            else:
                FP[detection_idx]=1
                
        TP_cumsum=torch.cumsum(TP,dim=0)
        FP_cumsum=torch.cumsum(FP,dim=0)
        recalls=TP_cumsum/(total_true_bboxes+eps)
        precisions=TP_cumsum/(TP_cumsum+FP_cumsum+eps)
        precisions=torch.cat((torch.tensor([1]),precisions))
        recalls=torch.cat((torch.tensor([0]),recalls)) 
        average_precision.append(torch.trapz(precisions,recalls))
        
          
    
    return sum(average_precision)/len(average_precision)