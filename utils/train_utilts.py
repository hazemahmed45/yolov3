import torch
from tqdm import tqdm


def train_loop(data_loader,model,optimizer,criterion,device,is_valid):
    iter_loop=tqdm(data_loader,total=len(data_loader))
    mean_loss=[]
    
    for ii,(batch_imgs,batch_labels) in iter_loop:
        batch_imgs,batch_labels=batch_imgs.to(device),batch_labels.to(device)
        out=model(batch_imgs)
        loss=criterion(out,batch_labels)
        mean_loss.append(loss.item())
        iter_loop.set_description(desc="Train" if not is_valid else 'Valid')
        iter_loop.set_postfix({'LOSS':sum(mean_loss)//len(mean_loss)})
        if(not is_valid):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
    return 

