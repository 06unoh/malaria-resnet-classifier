import torchvision.transforms as transforms

def get_train_tf(mean, std):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
def get_test_tf(mean, std):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
def get_temp_tf():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])