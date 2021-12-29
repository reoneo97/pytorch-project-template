from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class ImageDataset(Dataset):
    def __init__(self,):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def image_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return transform


def cifar10_dataloader(path: str, batch_size: int) -> DataLoader:
    ds = CIFAR10(path, download=True, transform=image_transform())
    dl = DataLoader(ds, batch_size, shuffle=True)
    return dl


cifar_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
