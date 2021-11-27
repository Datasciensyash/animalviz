import dataclasses
import typing as tp
from pathlib import Path

import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms


@dataclasses.dataclass
class ClipOutput:
    text_embedding: np.ndarray
    image_embedding: np.ndarray
    probability: float


class ClipInterface:
    def __init__(
            self,
            model_name: str = "RN101",
            device: str = 'cpu',
            num_trials: int = 10,
    ):
        self.device = device
        self.num_trials = num_trials
        self.model, self.transform = self._load_model(model_name)
        self.model.to(self.device)

    def prepare_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(str(image_path)).convert("RGB")
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def get_embedding(self, image_tensor: torch.Tensor):
        if image_tensor.shape[0] != 1:
            raise ValueError(
                f'This function can only calculate embedding for batch_size 1, '
                f'got {image_tensor.shape[0]}'
            )
        with torch.no_grad():
            embedding = self.model(image_tensor).flatten().cpu().numpy()
        return embedding

    def get_visual_embedding(self, image_path: Path) -> np.ndarray:
        with torch.no_grad():
            image_tensor = self.prepare_image(image_path)
            image_tensor = self.model.encode_image(image_tensor).float()
            image_tensor /= image_tensor.norm(dim=-1, keepdim=True)
            return image_tensor.detach().cpu().numpy()

    def get_textual_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokenized = clip.tokenize([text]).to(self.device)
            text_tensor = self.model.encode_text(tokenized).float()
            text_tensor /= text_tensor.norm(dim=-1, keepdim=True)
            return text_tensor.detach().cpu().numpy()

    def classify_image(self, image_path: Path, text: str) -> ClipOutput:
        outputs = []
        for _ in range(self.num_trials):
            image_embedding = self.get_visual_embedding(image_path)
            text_embedding = self.get_textual_embedding(text)
            probability = (text_embedding @ image_embedding.T).flatten()[0]
            outputs.append(
                ClipOutput(
                    image_embedding=image_embedding,
                    text_embedding=text_embedding,
                    probability=probability
                )
            )
        return max(outputs, key=lambda x: x.probability)

    @staticmethod
    def _load_model(
            model_name: str
    ) -> tp.Tuple[torch.nn.Module, torchvision.transforms.Compose]:
        model, preprocess = clip.load(model_name)
        preprocess.transforms = [transforms.Resize(512)] + preprocess.transforms
        preprocess.transforms[1] = transforms.RandomCrop(model.visual.input_resolution)
        model.eval()
        return model, preprocess
