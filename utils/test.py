import torch

def test(model, testloader, device):
    total=0
    correct=0    
    model.eval()
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels=imgs.to(device), labels.to(device)
            outputs=model(imgs)
            preds=torch.round(torch.sigmoid(outputs)).squeeze()
            total+=labels.size(0)      
            correct+=(preds==labels).sum().item()
    
    acc=(correct*100)/total
    print(f'Accuracy: {acc:.2f}%')    