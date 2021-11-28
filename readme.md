## AnimalViz
Tool for classification of different animals on photos.

## Installation

```
# Clone repo
git clone https://github.com/Datasciensyash/animalviz.git && cd animalviz

# Install requirements
pip install -r requirements.txt

# Install additional requirements
pip install git+https://github.com/openai/CLIP.git
```

## Run classification

```
python run.py

usage: run.py [-h] -p PATH [-o OUT_PATH] [-m MODELS_PATH] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to directory with .jpg files (default: None)
  -o OUT_PATH, --out_path OUT_PATH
                        Path to output .csv file (default: ./labels.csv)
  -m MODELS_PATH, --models_path MODELS_PATH
                        Path to directory with models. (default: ./model/)
  -d DEVICE, --device DEVICE
                        Computing device, e.g. cuda:0 or cpu (default: cpu)
```
Example - run classification on `./images/` directory on `cuda:0`:

```
python run.py -p ./images/ -d cuda:0
```

## Run visualization tool

```
streamlit run visualize.py
```

## Predictions for hack
File `labels.csv` stored in `./outputs/labels.csv`