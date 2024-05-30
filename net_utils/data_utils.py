from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 dataset
def load_CIFAR10(cifar10_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(cifar10_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(cifar10_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader