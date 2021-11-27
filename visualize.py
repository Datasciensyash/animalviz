from pathlib import Path

import streamlit as st
import yaml
from tqdm import tqdm

from deep_translator import GoogleTranslator

from animalviz.clip_interface import ClipInterface
from animalviz.eval_interface import ClassificationOutput
from animalviz.model_loading import load_from_directory

translator = GoogleTranslator(source='auto', target='en')

path_to_directory = st.sidebar.text_input('Path to directory')
files_to_predict = list(Path(path_to_directory).rglob('*.jpg'))
device = st.sidebar.text_input('Device', value='cpu')

task = st.sidebar.selectbox("Choose task", ["Classification", "Ranking"])

if task == "Classification":
    path_to_model = st.sidebar.text_input('Path to model', value="./models/")
    path_to_model = Path(path_to_model)

    eval_interface = load_from_directory(path_to_model, device)

    predictions = []
    progress_bar = st.sidebar.progress(0)
    for i, filename in tqdm(enumerate(files_to_predict)):
        progress_bar.progress(i / len(files_to_predict))

        meta_fname = filename.with_suffix('.yml')
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
        predictions.append((filename, cls_out))
    progress_bar.progress(1.)

    classes = set([p[1].label_name for p in predictions])
    cls_to_show = st.sidebar.selectbox("Select class to show", list(classes))
    num_to_show = st.sidebar.number_input("Number of images to show", min_value=1, max_value=10, value=5)
    predictions = [p for p in predictions if p[1].label_name == cls_to_show]
    predictions = sorted(predictions, key=lambda x: x[1].probability, reverse=True)
    for file, prediction in predictions:
        st.write(f'Predicted class {cls_to_show} with probability {prediction.probability}')
        st.image(str(file))

elif task == "Ranking":
    clip_interface = ClipInterface(device=device)
    top_n = st.sidebar.number_input("Number of images to show", min_value=1, max_value=10, value=5)

    russian_input = st.sidebar.text_input("What to find on images?", value="Лиса")
    english_input = translator.translate(russian_input)

    predictions = []
    progress_bar = st.sidebar.progress(0)
    for i, file in tqdm(enumerate(files_to_predict)):
        progress_bar.progress(i / len(files_to_predict))
        cls_out = clip_interface.classify_image(file, russian_input)
        predictions.append((file, cls_out))
    progress_bar.progress(1.)

    predictions = sorted(predictions, key=lambda x: x[1].probability, reverse=True)
    for i in range(min(top_n, len(predictions))):
        st.write(f'Probability of {russian_input}: {predictions[i][1].probability}')
        st.image(str(predictions[i][0]))