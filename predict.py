import importlib
import inspect
import os
import sys

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

model_classes = []


def parse_args():
    parser = ArgumentParser(add_help=True)

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
        required=True,
        type=str,
        help='Model file path, must be a (.?) file. If specified this model will be trained'
    )

    parser.add_argument(
        '-b', '--batch',
        default=128,
        type=int,
        help='Batch size during training. Default: 128'
    )

    parser.add_argument(
        '-ts', '--timesteps',
        default=10,
        type=int,
        help='The amount of time step into the future you predict. Default: 10'
    )

    parser.add_argument(
        '-tu', '--timeunit',
        default=['10', 'minutes'],
        type=[],
        required=True,
        help='Specify the timeunit difference between rows. Default: [10, minutes]'
    )

    for model in model_classes:
        parser.add_argument(model.name)
        model.add_arguments(parser)

    return parser.parse_args()


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


def main(args):
    path = str(Path(__file__).parent / args.data)
    df = pd.read_csv(path)

    created_models = []

    for model_dir in args.model:
        p = Path(__file__).parent / model_dir
        for file in os.listdir(p):
            if file.endswith('.txt') & file.startswith('metadata'):
                metadata = read_metadata(p / file)
            else:
                weights_file = p / file

        if metadata is not None:
            model_ = find_model(metadata[0])

            if model_ is None:
                print(f"Couldn't find model: {metadata[0]}")
                quit(101)
                break

            model_class_ = model_(metadata[1], df)

            compatability, missing_columns = check_compatability(df, metadata)

            if not compatability:
                print(
                    'One of the models is not compatible with the dataset and is missing: ' + str(missing_columns))
                quit(102)
                break

            created_models.append(model_class_)

    for model in created_models:
        dataset = model.generate_time_series_dataset(**vars(args))

        trained_model = model.load_model(weights_file, **vars(args))

        predictions = model.predict(trained_model, dataset, **vars(args))

        print(predictions)


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
        print(name_local)
        if inspect.isclass(getattr(c, name_local)):
            Model = getattr(c, name_local)
            model_classes.append(Model)
            print(f"{Model.name} model has been loaded in")

    main(parse_args(), models)

    # TODO:
    # -Let the user predict X amount of time-units into the future (see bullet point 5)
    # -Discuss what kind of output / prediction we want

    # -Specify Timestep (Bryan)
    # -Specify which is the date column
    # -Figure out why it only predicts 6 timesteps
    # -Move specified column to convert
    # -Generate metadata file for algoritme
    # -Make index more time-based (this will increase accuracy by a lot)
    # -Look into WANDB
