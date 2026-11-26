"""Shared model construction and deterministic inference preprocessing."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

PREPROCESSING = "letterbox224-imagenet-v1"


class Letterbox:
    def __call__(self, image):
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("Expected an RGB PIL image")
        if min(image.size) <= 0:
            raise ValueError("Empty image")
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR,
                            color=(124, 116, 104), centering=(0.5, 0.5))


def preprocessing(training=False):
    steps = [Letterbox()]
    if training:
        steps += [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                  transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.15)]
    return transforms.Compose(steps + [transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def make_model(pretrained=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def select_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process. No automatic CPU fallback; check GPU access.")
    if name not in ("mps", "cpu"):
        raise ValueError("Supported devices: mps, cpu")
    return torch.device(name)


class DriverClassifier:
    """Stage 2 stateless image classifier; temporal monitoring comes in stage 3."""
    def __init__(self, bundle_path, device="mps"):
        bundle = Path(bundle_path)
        self.metadata = json.loads((bundle / "bundle.json").read_text())
        if self.metadata["preprocessing"] != PREPROCESSING or self.metadata["architecture"] != "resnet18":
            raise ValueError("Unsupported bundle architecture or preprocessing")
        payload = (bundle / "model.pt").read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.metadata["weights_sha256"]:
            raise ValueError("Model weights checksum mismatch")
        self.device = select_device(device)
        self.model = make_model()
        self.model.load_state_dict(torch.load(bundle / "model.pt", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.transform = preprocessing()

    @torch.inference_mode()
    def predict(self, images):
        if not images:
            raise ValueError("At least one image is required")
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        probabilities = self.model(batch).softmax(1).cpu()
        return [{"predicted_class": int(p.argmax()),
                 "class_name": self.metadata["class_names"][int(p.argmax())],
                 "probabilities": p.tolist(), "frame_score": float(1 - p[0]),
                 "model_version": self.metadata["model_version"]} for p in probabilities]
