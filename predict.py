"""
A prediction file that can use multiple models for predicting the forecasting

command-arguments:
    -d [--DATA]: Filepath to data source
    -t [--TARGETS]: columns of the data that need to be predicted
    -g [--GROUPS]: groups in datasets
    -kr [--KNREELS]: Known reels features in dataset (Also known in the future). Default: []
    -kc [--KNCATS]: Known categorical features in dataset (also know in the future). Default: []
    -ur [--UNREELS]: Unknown reel features in dataset. Default: []
    -uc [--UNCATS]: Unknown categorical features in dataset. Default: []
    -m [--MODEL]: Model file paths, must be a file.
    -ts [--TIMESTEPS]: The amount of timesteps into the future you predict. Default: 10
    -tu [--TIMEUNIT]: Specify the timeunit difference between rows. Default: [10, min]

"""

import importlib
import inspect
import os
import sys

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

model_classes = []


def parse_args():
    """
    Parse the arguments from the command using argparse

    :param models: the models being used by the command

    :return: argument-parser
    """

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
        help='Model file paths, must be a (.?) file.'
    )

    parser.add_argument(
        '-ts', '--timesteps',
        default=10,
        type=int,
        help='The amount of time step into the future you predict. Default: 10'
    )

    parser.add_argument(
        '-tu', '--timeunit',
        action='extend',
        default=['10', 'min'],
        nargs='*',
        required=True,
        help='Specify the timeunit difference between rows. Default: [10, minutes]'
    )

    for model in model_classes:
        parser.add_argument(model.name)
        model.add_arguments(parser)

    return parser.parse_args()


def read_metadata(file):
    """
    Reads the metadata file that was generated when the model was trained

    :param file: location of the file
    :return: [model_name, model_id, data_source, targets, column_names]
    """
    f = open(file, 'r+')

    lines = f.readlines()

    return [lines[0].strip(), lines[1].strip(), lines[2], eval(lines[3]), eval(lines[4])]


def find_model(name):
    """
    Finds the model based on name. If none exists it will return None

    :param name: model_name
    :return:
    """
    for model in model_classes:
        if str(model.name) == str(name):
            return model


def check_compatability(df, metadata):
    """
    Checks whether or not the model is can be used to predict a forecast based on the data that was provided

    :param df: Dataframe that is loaded in for predicting or training
    :param metadata: metadata that was generated for model
    :return: Boolean, [missing_columns]
    """
    columns = list(df.columns.values)

    missing_columns = []

    compatability = True

    for column in metadata[4]:
        if not columns.__contains__(column):
            missing_columns.append(column)
            compatability = False

    return compatability, missing_columns


def main(args):
    """
    The main method that runs whenever the file is being used.

    :param args: the arguments in the command.
    :param chosen_models: the models have been specified in the command

    This method loops through the chosen models and executes

    Model.load_model
    Model.predict

    :return: Nothing
    """
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
        if inspect.isclass(getattr(c, name_local)):
            Model = getattr(c, name_local)
            model_classes.append(Model)
            print(f"{Model.name} model has been loaded in")

    main(parse_args())

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
