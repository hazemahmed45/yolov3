import torch
import os

from loss import YoloLoss
from model import YOLOv3
from tqdm import tqdm
import config
from torch.optim import Adam
from utils import (
    cells_to_bboxes,
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)

torch.backends.cudnn.benchmark=True

def train_loop(train_loader,model,optimizer,criterion,scaler,scaled_anchors):
    loop=tqdm(train_loader,leave=True)
    losses=[]
    for ii,(batch_imgs,batch_targets) in enumerate(loop):
        batch_imgs=batch_imgs.to(config.DEVICE)
        batch_target_s0,batch_targets_s1,batch_targets_s2=(
            batch_targets[0].to(config.DEVICE),
            batch_targets[1].to(config.DEVICE),
            batch_targets[2].to(config.DEVICE)
        )
        with torch.cuda.amp.autocast_mode.autocast():
            out = model(batch_imgs)
            loss=criterion(out[0],batch_target_s0,scaled_anchors[0])+\
                criterion(out[1],batch_targets_s1,scaled_anchors[1])+\
                criterion(out[2],batch_targets_s2,scaled_anchors[2])
        
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=sum(losses)/len(losses))
    return

def main():
    model=YOLOv3(config=config.ARCHI,num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer=Adam(model.parameters(),lr=config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
    train_loader,test_loader,train_eval_loader=get_loaders(
        os.path.join(config.DATASET,'8examples.csv'),
        test_csv_path=os.path.join(config.DATASET,'test.csv')
    )
    
    if(config.LOAD_MODEL):
        load_checkpoint(config.CHECKPOINT_FILE,model,optimizer,config.LEARNING_RATE)
    scaled_anchors=(torch.tensor(config.ANCHORS)*torch.tensor(config.S).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,3,2)).to(config.DEVICE)
    criterion=YoloLoss()
    scaler=torch.cuda.amp.grad_scaler.GradScaler()
    
    
    for e in range(config.NUM_EPOCHS):
        train_loop(train_loader,model,optimizer,criterion,scaler,scaled_anchors)
        if(config.SAVE_MODEL):
            save_checkpoint(model,optimizer,config.CHECKPOINT_FILE)
        if(e%10==0 and e>0):
            print("Testing...")
            check_class_accuracy(model,test_loader,config.CONF_THRESHOLD)
            
            pred_bboxes,true_bboxes=get_evaluation_bboxes(test_loader,model,config.NMS_IOU_THRESH,config.ANCHORS,config.CONF_THRESHOLD)
            
            map_val=mean_average_precision(pred_bboxes,true_bboxes,config.MAP_IOU_THRESH,'midpoint',config.NUM_CLASSES)
            print("MAP: ",map_val.item())
            
    return 
if(__name__ == '__main__'):
    main()