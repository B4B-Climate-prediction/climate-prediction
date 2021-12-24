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
import matplotlib.pyplot as plt

from utils import config_reader

model_classes = []


def parse_args():
    """
    Parse the arguments from the command using argparse

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
        '-m', '--model',
        action='extend',
        nargs='*',
        default=[],
        help='Model file paths, must be a (.?) file. If specified this model will be trained'
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
        default=['10', 'minutes'],
        nargs='*',
        required=True,
        help='Specify the timeunit difference between rows. Default: [10, minutes]'
    )

    return parser.parse_args()


def check_compatability(df, columns):
    """
    Checks whether or not the model is can be used to predict a forecast based on the data that was provided

    :param df: Dataframe that is loaded in for predicting or training
    :param columns: columns in the metadata that was generated for model
    :return: Boolean, [missing_columns]
    """
    df_columns = list(df.columns.values)

    missing_columns = []

    compatability = True

    for column in columns:
        if not df_columns.__contains__(column):
            missing_columns.append(column)
            compatability = False

    return compatability, missing_columns


def find_model(name):
    """
    Finds the model based on name. If none exists it will return None

    :param name: model_name
    :return:
    """
    for model in model_classes:
        if str(model.name) == str(name):
            return model


def plot_predictions(observed, predicted):
    plt.plot(observed, label="Observed")
    plt.plot(predicted, label="Predicted")
    plt.legend()
    plt.show()


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
    global metadata, weights_file
    df_path = str(Path(__file__).parent / 'out' / 'datasets' / args.data)
    df = pd.read_csv(df_path, parse_dates=['Timestamp'])

    created_models = []
    files = []

    for model_dir in args.model:
        p = Path(__file__).parent / model_dir
        for file in os.listdir(p):
            if file.endswith('.cfg') & file.startswith('metadata'):
                metadata = config_reader.read_metadata(p / file, loaded_models=model_classes)
            else:
                weights_file = p / file

        if metadata is not None:
            model_ = find_model(metadata['model'])

            if model_ is None:
                print(f"Couldn't find model: {metadata['model']}")
                quit(101)
                break

            model_class_ = model_(metadata['id'], metadata, df)

            compatability, missing_columns = check_compatability(df, metadata['columns'])

            if not compatability:
                print(
                    'One of the models is not compatible with the dataset and is missing: ' + str(missing_columns))
                quit(102)
                break

            if weights_file is None:
                print(f'Cannot find weights file: {metadata["id"]}')
                quit(104)
                break

            created_models.append(model_class_)
            files.append(weights_file)

        else:
            print(f"Metadata file is invalid. Directory: {model_dir}")
            quit(103)
            break

    for index in range(0, len(created_models)):
        file = files[index]
        model = created_models[index]
        trained_model = model.load_model(file, **vars(args))

        column_observed_name = model.metadata['targets'][0]
        column_prediction_name = f'{model.metadata["id"]}-{model.metadata["targets"][0]}'

        predictions = model.predict(trained_model, **vars(args))
        df[column_prediction_name] = predictions

        plot_predictions(
            observed=df[column_observed_name],
            predicted=df[column_prediction_name]
        )


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
    # -Figure out why it only predicts 6 timesteps (this is because min- and max_prediction length in the dataset are both set to 6)
    # -Move specified column to convert
    # -Generate metadata file for algoritme
    # -Make index more time-based (this will increase accuracy by a lot)
    # -Look into WANDB
