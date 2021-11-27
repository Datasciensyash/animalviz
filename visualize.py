from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from tqdm import tqdm

from animalviz.eval_interface import ClassificationOutput, EvalInterface
from animalviz.model_loading import load_from_directory


path_to_model = st.sidebar.text_input('Path to model', value="./models/")
path_to_model = Path(path_to_model)

device = st.sidebar.text_input('Device', value='cpu')

path_to_directory = st.sidebar.text_input('Path to directory')
files_to_predict = list(Path(path_to_directory).rglob('*.jpg'))

eval_interface = load_from_directory(path_to_model, device)

predictions = []

progress_bar = st.sidebar.progress(0)
for i, file in tqdm(enumerate(files_to_predict)):
    progress_bar.progress(i / len(files_to_predict))

    meta_fname = file.with_suffix('.yml')
    if not meta_fname.exists():
        cls_out = eval_interface.classify_image(file)
        with meta_fname.open('w') as file:
            yaml.dump({
                'label': cls_out.label,
                'label_name': cls_out.label_name,
                'probability': cls_out.probability
            }, file)
    else:
        with meta_fname.open('r') as file:
            meta = yaml.load(file, Loader=yaml.SafeLoader)
            cls_out = ClassificationOutput(
                **meta
            )
    predictions.append(cls_out)
progress_bar.progress(1.)

st.write(predictions)