import torch

def get_mean_std(dataloader):
    mean=torch.zeros(3)
    std=torch.zeros(3)
    num_sample=0
    
    for img,_ in dataloader:
        batch=img.size(0)
        img=img.view(batch,3,-1)
        mean+=img.mean(dim=[0,2])*batch
        std+=img.std(dim=[0,2])*batch
        num_sample+=batch
    mean=mean/num_sample
    std=std/num_sample

    return mean, std


