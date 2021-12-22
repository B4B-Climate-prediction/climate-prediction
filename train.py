"""
A training file that can take multiple models, train them and evaluate them.

Command-arguments:
    -d [--DATA]: Data file path, must be a .csv file.
    -m [--MODEL]: Model file paths, must be a file. If specified it will continue training that model.
"""

import importlib
import inspect
import os
import uuid
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger

from utils import config_reader

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
        '-m', '--model',
        action='extend',
        nargs='*',
        default=[],
        help='Model file paths, must be a (.?) file. If specified this model will be trained'
    )

    return parser.parse_args()


def find_model(name):
    """
    Finds the model based on name. If none exists it will return None

    :param name: model_name
    :return:
    """
    for model in model_classes:
        if str(model.name) == str(name):
            return model


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


def main(args):
    """
    The main method that runs whenever the file is being used.

    :param args: the arguments in the command.

    This method loops through the chosen models and executes

    Model.generate_time_series_dataset

    if hypertuning is enabled it will call:

        -Model.tune_hyper_parameter

    else it will check for an existing model otherwise it will generate a new model.
        -Model.load_model
        -Model.generate_model

    -Model.train_model
    -Model.evaluate_model

    :return: Nothing
    """

    # TODO Fix WandB logger! Move this bs
    global metadata, weights_file

    path = str(Path(__file__).parent / args.data)
    df = pd.read_csv(path)

    if len(args.model) != 0:

        created_models = []
        files = []

        for model_dir in args.model:
            p = Path(__file__).parent.absolute() / model_dir
            for file in os.listdir(p):
                if file.endswith('.cfg') & file.startswith('metadata'):
                    metadata = config_reader.read_metadata(p / file)
                else:
                    weights_file = p / file

            if metadata is not None:
                model = find_model(metadata['model'])

                if model is None:
                    print(f"Couldn't find model: {metadata['model']}")
                    quit(101)
                    break

                model_class = model(metadata['id'], df)

                compatability, missing_columns = check_compatability(df, metadata['columns'])

                if not compatability:
                    print(
                        'One of the models is not compatible with the dataset and is missing: ' + str(missing_columns))
                    quit(102)
                    break

                if weights_file is not None:
                    print(f'Cannot find model file of model: {metadata["name"]}')
                    quit(104)
                    break

                created_models.append(model_class)
                files.append(weights_file)
            else:
                print(f"Cannot load in metadata. Directory: {model_dir}")
                quit(103)
                break

        for index in range(0, len(created_models) - 1):
            model = created_models[index]
            file = files[index]

            training = model.generate_time_series_dataset(**vars(args))

            trained_model = model.load_model(file, **vars(args))

            os.remove(file)

            trained_model = model.train_model(training, trained_model, **vars(args))

            # model.evaluate_model(trained_model, training, **vars(args))

    else:
        # ToDo: MOVE THIS BS TO CONFIG FILE!
        configs = config_reader.read_configs(Path(__file__).parent.absolute() / 'configs', loaded_models=model_classes)

        for config in configs:
            model = find_model(config['model'])

            if model is not None:
                model_id = uuid.uuid4()

                model_class = model(model_id, df)

                training = model_class.generate_time_series_dataset(**vars(config))

                if config['hyper']:
                    trained_model = model_class.tune_hyper_parameter(df, **vars(config))
                else:
                    c_model = model_class.generate_model(training, **vars(config))

                    trained_model = model_class.train_model(training, c_model, **vars(config))

                config_reader.export_metadata(config['model'], model_id, df,
                                              Path(__file__).parent.absolute() / 'out' / 'models' / f'{config["model"]}' / f'{model_id}' / 'checkpoints')

                model_class.evaluate_model(trained_model, **vars(args))
            else:
                print(f"Couldn't find model: {config['model']}")
                quit(102)
                break


if __name__ == '__main__':
    # search models in folder
    package_dir = Path(__file__).parent / 'models'

    c = importlib.import_module(f"{package_dir}")

    for name_local in dir(c):
        if inspect.isclass(getattr(c, name_local)):
            Model = getattr(c, name_local)
            model_classes.append(Model)
            print(f"{Model.name} model has been loaded in")

    main(parse_args())

# Check names of the loaded models with the corresponding folder
# Load metadata into train.py and predict.py for checking
# Checking the folder that contains the trained model/weights .cktp
