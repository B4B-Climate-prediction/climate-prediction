"""
A training file that can take multiple models, train them and evaluate them.

Command-arguments:
    -w [--WANDB]: API Key for WandB
    -d [--DATA]: Data file path, must be a .csv file.
    -t [--TARGETS]: columns of the data that need to be predicted
    -g [--GROUPS]: groups in datasets
    -kr [--KNREELS]: Known reels features in dataset (Also known in the future). Default: []
    -kc [--KNCATS]: Known categorical features in dataset (also know in the future). Default: []
    -ur [--UNREELS]: Unknown reel features in dataset. Default: []
    -uc [--UNCATS]: Unknown categorical features in dataset. Default: []
    -m [--MODEL]: Model file paths, must be a file. If specified it will continue training that model.
    -hy [--HYPER]: hypertunes the model if specified, this will take a long time.
    -e [--EPOCHS]: Amount of epochs to train. Default: 100
    -b [--BATCH]: Batch size during training. Default: 128
    -tr [--TRAILS]: The amount of times the hypertuning generates new hyper parameters to train the model. Default: 100
"""

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
    """
    Parse the arguments from the command using argparse

    :param models: the models being used by the command

    :return: argument-parser
    """
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
        help='Model file paths, must be a (.?) file. If specified this model will be trained'
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
    """
    Generation of metadata file for the models

    :param model_name: name of the model
    :param model_id: model id
    :param args: arguments from command
    :param pl: saving_path
    :return: [model_name, id, data_source, targets, column_name]
    """

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
    """
   Reads the metadata file that was generated when the model was trained

   :param file: location of the file
   :return: [model_name, model_id, data_source, targets, column_names]
   """
    
    with open(file, 'r+') as f:
        lines = f.readlines()
        f.close()

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


def main(args, chosen_models):
    """
    The main method that runs whenever the file is being used.

    :param args: the arguments in the command.
    :param chosen_models: the models have been specified in the command

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

    # TODO Fix WandB logger!
    global metadata, weights_file
    logger = None
    if args.wandb is not None:
        team = 'b4b-cp'
        project = 'climate-prediction'

        logger = WandbLogger(project=project)
        os.environ['WANDB_API_KEY'] = args.wandb

        wandb.init(project=project, entity=team)
        wandb.login()

    df_path = str(Path(__file__).parent / 'out' / 'datasets' / args.data)
    df = pd.read_csv(df_path, parse_dates=['Timestamp'])

    if len(args.model) != 0:

        created_models = []
        files = []

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

                if weights_file is not None:
                    print(f'Cannot find model file of model: {metadata[0]}')
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

                export_metadata(name, model_id, args,
                                (Path(__file__).parent / 'out' / 'models' / f'{name}' / f'{model_id}' / 'checkpoints'))

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
