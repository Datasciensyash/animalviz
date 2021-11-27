import dataclasses
import typing as tp
from pathlib import Path

import cv2
import lightgbm
import numpy as np
import torch
from torchvision import models, transforms


@dataclasses.dataclass
class ClassificationOutput:
    label: int
    label_name: str
    probability: float


class EvalInterface:
    def __init__(
            self, model_name: str,
            lgb_classifier_path: Path,
            labels: tp.Dict[int, str],
            weights: np.ndarray,
            image_size: int = 768,
            device: str = 'cpu'
    ):
        self.device = device
        self.model = self._load_model(model_name)
        self.classifier = self._load_pretrained_lgb(lgb_classifier_path)
        self.labels = labels
        self.image_size = image_size
        self.weights = weights

        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(image_size)
        ])

    def prepare_image(self, image_path: Path) -> torch.Tensor:
        image_tensor = cv2.imread(str(image_path))
        image_tensor = torch.from_numpy(image_tensor).to(self.device)
        image_tensor = image_tensor.transpose(-1, 0) / 255.
        image_tensor = self.transform(image_tensor)
        return image_tensor.unsqueeze(0)

    def get_embedding(self, image_tensor: torch.Tensor):
        if image_tensor.shape[0] != 1:
            raise ValueError(
                f'This function can only calculate embedding for batch_size 1, '
                f'got {image_tensor.shape[0]}'
            )
        with torch.no_grad():
            embedding = self.model(image_tensor).flatten().cpu().numpy()
        return embedding

    def classify_embedding(self, embedding: torch.Tensor) -> tp.Tuple[int, float]:
        predictions = self.classifier.predict([embedding])[0]
        predictions = predictions * self.weights
        class_label = predictions.argmax()
        class_proba = predictions[class_label]
        return int(class_label), float(class_proba)

    def classify_image(self, image_path: Path) -> ClassificationOutput:
        image = self.prepare_image(image_path)
        embedding = self.get_embedding(image)
        cls_label, cls_proba = self.classify_embedding(embedding)
        return ClassificationOutput(
            label=cls_label,
            label_name=self.labels[cls_label],
            probability=cls_proba
        )

    @staticmethod
    def _load_pretrained_lgb(path: Path) -> lightgbm.Booster:
        return lightgbm.Booster(model_file=str(path))

    def _load_model(self, model_name: str) -> torch.nn.Module:
        model = getattr(models, model_name)(pretrained=True).to(self.device)
        model.eval()
        return torch.nn.Sequential(*list(model.children())[:-1])

