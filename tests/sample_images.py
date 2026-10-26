"""Small natural test images without downloading anything.

scikit-learn ships two photos (china.jpg, flower.jpg); we cut 224x224 crops from them.
"""
import torch
import torchvision.transforms.functional as TF


def sample_batch(n: int = 8, size: int = 224) -> torch.Tensor:
    from sklearn.datasets import load_sample_images
    photos = [TF.to_tensor(img.copy()) for img in load_sample_images().images]
    crops = []
    for photo in photos:
        _, h, w = photo.shape
        for top in range(0, h - size + 1, 40):
            for left in range(0, w - size + 1, 80):
                crops.append(photo[:, top:top + size, left:left + size])
    if n > len(crops):
        raise ValueError(f"only {len(crops)} sample crops available")
    return torch.stack(crops[:n])
