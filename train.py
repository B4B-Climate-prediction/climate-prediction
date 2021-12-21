import importlib
import inspect
import sys
import uuid

import wandb
import os
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime

from pathlib import Path
import model as ml

from pytorch_lightning.loggers import WandbLogger

model_classes = []


def parse_args():
    parser = ArgumentParser(add_help=True)

    parser.add_argument(
        '-w', '--wandb',
        type=str,
        help='API key for WandB'
    )

    parser.add_argument(
        '-d', '--data',
        required=True,
        type=str,
        help='Data file path, must be a .csv file in the out/datasets directory.'
    )

    parser.add_argument(
        '-t', '--targets',
        required=True,
        action='extend',
        nargs='+',
        help='Target feature(s) in dataset, specify minimal one'
    )

    parser.add_argument(
        '-g', '--groups',
        required=True,
        action='extend',
        nargs='+',
        help='Groups in dataset, specify minimal one'
    )

    parser.add_argument(
        '-kr', '--knreels',
        action='extend',
        nargs='*',
        default=[],
        help='Known reels features in dataset (also known in the future). Default: []'
    )

    parser.add_argument(
        '-kc', '--kncats',
        action='extend',
        nargs='*',
        default=[],
        help='Known categoricals features in dataset (also known in the future). Default: []'
    )

    parser.add_argument(
        '-ur', '--unreels',
        action='extend',
        nargs='*',
        default=[],
        help='Unknown reels features in dataset. Default: []'
    )

    parser.add_argument(
        '-uc', '--uncats',
        action='extend',
        nargs='*',
        default=[],
        help='Unknown categoricals features in dataset. Default: []'
    )

    parser.add_argument(
        '-m', '--model',
        action='extend',
        nargs='*',
        default=[],
        help='Model file path, must be a (.?) file. If specified this model will be trained'
    )

    parser.add_argument(
        '-hy', '--hyper',
        action='store_true',
        help='Hypertunes the model if specified, this will take a long time'
    )

    parser.add_argument(
        '-e', '--epochs',
        default=100,
        type=int,
        help='Amount of epochs to training. Default: 100'
    )

    parser.add_argument(
        '-b', '--batch',
        default=128,
        type=int,
        help='Batch size during training. Default: 128'
    )

    parser.add_argument(
        '-tr', '--trials',
        default=100,
        type=int,
        help='Trials should only be used whenever hypertuning is activated. Trials specifies the amount of trials it '
             'will maximally run before coming to the best results. Default: 100 '
    )

    for model in model_classes:
        parser.add_argument(model.name)
        model.add_arguments(parser)

    return parser.parse_args()


def export_metadata(model_name, model_id, args, pl):
    # Retrieve dataset
    path = str(Path(__file__).parent / args.data)
    df = pd.read_csv(path)

    # Get columns names from dataset
    column_names = list(df.columns.values)

    metadata = [model_name, model_id, args.data, args.targets, column_names]

    # Create name for metadata file
    time = datetime.now().strftime('%H%M%S')
    name = f'metadata-{time}-{args.epochs}.txt'

    export_path = Path(pl) / name

    f = open(export_path, 'w+')

    for meta in metadata:
        f.write(str(meta) + '\n')

    f.close()


def read_metadata(file):
    f = open(file, 'r+')

    lines = f.readlines()

    return [lines[0].strip(), lines[1].strip(), lines[2], eval(lines[3]), eval(lines[4])]


def find_model(name):
    for model in model_classes:
        if str(model.name) == str(name):
            return model


def check_compatability(df, metadata):
    columns = list(df.columns.values)

    missing_columns = []

    compatability = True

    for column in metadata[4]:
        if not columns.__contains__(column):
            missing_columns.append(column)
            compatability = False

    return compatability, missing_columns


def main(args, chosen_models):
    logger = None
    if args.wandb is not None:
        team = 'b4b-cp'
        project = 'climate-prediction'

        logger = WandbLogger(project=project)
        os.environ['WANDB_API_KEY'] = args.wandb

        wandb.init(project=project, entity=team)
        wandb.login()

    path = str(Path(__file__).parent / args.data)
    df = pd.read_csv(path)

    if len(args.model) != 0:

        created_models = []

        for model_dir in args.model:
            p = Path(__file__).parent / model_dir
            for file in os.listdir(p):
                if file.endswith('.txt') & file.startswith('metadata'):
                    metadata = read_metadata(p / file)
                else:
                    weights_file = p / file

            if metadata is not None:
                model = find_model(metadata[0])

                if model is None:
                    print(f"Couldn't find model: {metadata[0]}")
                    quit(101)
                    break

                model_class = model(metadata[1], df)

                compatability, missing_columns = check_compatability(df, metadata)

                if not compatability:
                    print(
                        'One of the models is not compatible with the dataset and is missing: ' + str(missing_columns))
                    quit(102)
                    break

                created_models.append(model_class)

            for model in created_models:
                training = model.generate_time_series_dataset(**vars(args))

                trained_model = model.load_model(weights_file, **vars(args))

                os.remove(weights_file)

                trained_model = model.train_model(training, trained_model, **vars(args))

                #model.evaluate_model(trained_model, training, **vars(args))

    else:
        for name in chosen_models:
            model = find_model(name)

            if model is not None:
                model_id = uuid.uuid4()

                model_class = model(model_id, df)

                training = model_class.generate_time_series_dataset(**vars(args))

                if args.hyper:
                    trained_model = model_class.tune_hyper_parameter(df, **vars(args))
                else:
                    c_model = model_class.generate_model(training, **vars(args))

                    trained_model = model_class.train_model(training, c_model, **vars(args))

                export_metadata(name, model_id, args, (Path(__file__).parent / 'out' / 'models' / f'{name}' / f'{model_id}' / 'checkpoints'))

                model_class.evaluate_model(trained_model, **vars(args))
            else:
                print(f"Couldn't find model: {name}")
                quit(102)
                break


if __name__ == '__main__':
    models = []

    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]

        if arg.startswith('-') | arg.startswith('--'):
            break

        models.append(arg)

    # search models in folder
    package_dir = Path(__file__).parent / 'models'

    c = importlib.import_module(f"{package_dir}")

    for name_local in dir(c):
        if inspect.isclass(getattr(c, name_local)):
            Model = getattr(c, name_local)
            model_classes.append(Model)
            print(f"{Model.name} model has been loaded in")

    main(parse_args(), models)

# Check names of the loaded models with the corresponding folder
# Load metadata into train.py and predict.py for checking
# Checking the folder that contains the trained model/weights .cktp
