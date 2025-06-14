import torch
import matplotlib.pyplot as plt

def visualize_prediction(model, testloader, device, std, mean):    
    dataiter=iter(testloader)
    imgs, labels=next(dataiter)
    imgs, labels=imgs.to(device), labels.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs=model(imgs)
        preds=torch.round(torch.sigmoid(outputs)).suqueeze()
        
    figs, axes=plt.subplots(3, 3, figsize=(8, 8))
    axes=axes.flatten()
    
    for i in range(9):
        img=imgs[i].cpu().permute(1,2,0).numpy()
        img=img*std+mean
        img=img.clip(0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f'Pred: {preds[i].item()}, Label: {labels[i].item()}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
