import argparse
import csv
from pathlib import Path
from tqdm import tqdm

from animalviz.model_loading import load_from_directory


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-p", "--path", help="Path to directory with .jpg files", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-o", "--out_path", help="Path to output .csv file", type=Path, required=False,
        default="./labels.csv"
    )
    arguments_parser.add_argument(
        "-m", "--models_path", help="Path to directory with models.", type=Path, required=False,
        default='./model/'
    )
    arguments_parser.add_argument(
        "-d", "--device", help="Path to directory with models.", type=Path, required=False,
        default='cpu'
    )
    args, unknown = arguments_parser.parse_known_args()
    return args


def main(models_path: Path, data_path: Path, out_path: Path, device: str):

    # Load model
    eval_interface = load_from_directory(models_path, device)

    # Prepare .csv file writer
    csvfile = out_path.open('w')
    writer = csv.writer(csvfile)

    # Define label_to_cls
    output_format = {'tiger': 1, 'leopard': 2, 'other': 3}

    # Inference
    for filename in tqdm(data_path.glob('*.jpg'), desc='Model inference...'):
        prediction = eval_interface.classify_image(filename)
        label = output_format[prediction.label_name]
        writer.writerow([filename.name, label])
    csvfile.close()

    print('Inference is successfully completed.')
    print(f'Output file path: {out_path.absolute()}')


if __name__ == '__main__':
    args = parse_args()
    main(
        models_path=args.models_path,
        data_path=args.path,
        out_path=args.out_path,
        device=args.device
    )
