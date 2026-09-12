"""
CIFAR-100 loading with 90/10 train/val split, indexed train loader
(for cache lookup), and unshuffled val/test loaders.
"""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .config import config
from .logger import SimpleLogger  # noqa: F401  (re-exported for convenience)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class IndexedDataset(Dataset):
    """Wraps a dataset so each item is (x, y, index)."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


def build_loaders(input_dir=None):
    """Construct train/val/test loaders and return (train, val, test, classes)."""
    test_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224, padding=8, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    try:
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.data_dir, train=False, download=True, transform=test_transform
        )
        print("✓ CIFAR-100 test set loaded")
    except Exception as e:
        print(f"⚠️ Error loading: {e}")
        input_data_dir = None
        if input_dir:
            input_data_dir = f"{input_dir}/data/cifar-100-python"
        if input_data_dir and torchvision.datasets.utils.check_integrity(input_data_dir) is False:
            pass
        test_dataset = torchvision.datasets.CIFAR100(
            root=input_dir, train=False, download=False, transform=test_transform
        )
        print("✓ CIFAR-100 loaded from input data")

    train_dataset = torchvision.datasets.CIFAR100(
        root=config.data_dir, train=True, download=True, transform=train_transform
    )
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(
        IndexedDataset(train_dataset),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    try:
        classes = test_dataset.classes
    except AttributeError:
        classes = [f'Class_{i}' for i in range(100)]

    return train_loader, val_loader, test_loader, classes


def evaluate(model, loader, criterion, config):
    """Evaluate model; return (avg_loss, acc, f1, preds, labels)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(config.device), labels.to(config.device)
            with autocast(enabled=config.use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1, all_preds, all_labels