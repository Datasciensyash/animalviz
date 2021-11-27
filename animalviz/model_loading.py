from pathlib import Path

import numpy as np
import yaml

from animalviz.eval_interface import EvalInterface


def load_from_directory(model_path: Path, device: str) -> EvalInterface:
    with (model_path / "meta.yml").open('r') as file:
        meta = yaml.load(file, Loader=yaml.SafeLoader)

    # Initialize model
    eval_interface = EvalInterface(
        model_name=meta["model_name"],
        lgb_classifier_path=(model_path / "model.txt"),
        labels=meta["labels"],
        weights=np.array(meta["weights"]),
        image_size=meta["image_size"],
        device=device
    )

    return eval_interface
