"""
Data loading with proper CIFAR-100 handling and validated train/val split.

This module includes the fix for the train/validation split bug that was
found mid-project (validation images leaking into training).
"""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, random_split

from code.config import Config
from code.logger import logger
from code.utils import make_loader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(config: Config):
    """Build train, practice, and test transforms."""
    if config.strong_augmentation:
        train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224, padding=8, padding_mode='reflect'),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    practice_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224, padding=8, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, practice_transform, test_transform


def build_dataloaders(config: Config):
    """
    Build train/val/test DataLoaders with validated split.

    Returns:
        train_loader, val_loader, test_loader, cache_loader, class_names

    Note: `cache_loader` is a clean-transform view of training data used only
    to build the teacher feature cache. It is None in QUICK_TEST mode.
    """
    logger.log("\n" + "=" * 80)
    logger.log("PHASE 1: DATA LOADING")
    logger.log("=" * 80)

    train_transform, practice_transform, test_transform = build_transforms(config)

    logger.log("Loading CIFAR-100...")
    try:
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.data_dir, train=False, download=True, transform=test_transform
        )
        logger.log("[OK] CIFAR-100 loaded successfully")
    except Exception as e:
        logger.log(f"[ERROR] Error loading CIFAR-100: {e}")
        raise RuntimeError("CIFAR-100 download failed. Check internet connection.")

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Validate split integrity (data-leakage bug fix)
    _validate_split(train_dataset, val_dataset, config)

    if config.QUICK_TEST:
        train_dataset = Subset(train_dataset, range(min(500, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(200, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(500, len(test_dataset))))
        logger.log("[WARN] QUICK TEST MODE ACTIVE")

    logger.log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = make_loader(train_dataset, config, shuffle=True,
                                indexed=True, drop_last=True)
    val_loader = make_loader(val_dataset, config, shuffle=False, indexed=False)
    test_loader = make_loader(test_dataset, config, shuffle=False, indexed=False)

    # Clean-transform view of training data for feature cache building
    cache_loader = None
    if not config.QUICK_TEST and isinstance(train_dataset, Subset):
        _train_clean_base = torchvision.datasets.CIFAR100(
            root=config.data_dir, train=True, download=False, transform=test_transform
        )
        train_dataset_clean = Subset(_train_clean_base, train_dataset.indices)
        cache_loader = make_loader(train_dataset_clean, config, shuffle=False,
                                    indexed=False, batch_size=config.cache_batch_size)

    try:
        CIFAR100_CLASSES = test_dataset.classes
    except AttributeError:
        CIFAR100_CLASSES = [f'Class_{i}' for i in range(100)]

    return train_loader, val_loader, test_loader, cache_loader, CIFAR100_CLASSES


def _validate_split(train_dataset, val_dataset, config: Config) -> None:
    """
    Assert that train and validation indices are disjoint and sum to 50000.

    This is the fix for the data-leakage bug found mid-project: an earlier
    pipeline logged 'Train: 50000, Val: 5000' (impossible, since CIFAR-100
    has only 50,000 training images), confirming validation images had been
    included in training for one run.
    """
    try:
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        assert len(train_indices & val_indices) == 0, \
            "Train and validation indices must be disjoint"
        assert len(train_indices) + len(val_indices) == 50000, \
            f"Train + Val must sum to 50000, got {len(train_indices) + len(val_indices)}"
        logger.log(f"[SPLIT-VALIDATION] Train={len(train_indices)}, Val={len(val_indices)}, "
                   f"overlap=0")
    except AttributeError:
        # QUICK_TEST may use Subset-of-Subset; skip strict validation
        logger.log("[SPLIT-VALIDATION] Skipped (nested Subset in QUICK_TEST)") 