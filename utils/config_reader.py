import os
from collections import ChainMap
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path


def read_configs(path, loaded_models) -> []:
    parser = ConfigParser()
    configs = []

    for cfg in os.listdir(path):
        parser.read(path / cfg)
        config = {
            'model': parser.get('model', 'name'),
            'targets': eval(parser.get('data', 'targets')),
            'groups': eval(parser.get('data', 'groups')),
            'unreels': eval(parser.get('data', 'unknown_reels')),
            'uncats': eval(parser.get('data', 'unknown_categoricals')),
            'knreels': eval(parser.get('data', 'known_reels')),
            'kncats': eval(parser.get('data', 'known_categoricals')),
            'hyper': eval(parser.get('hyper-tuning', 'hyper')),
            'trials': eval(parser.get('hyper-tuning', 'trials')),
            'batch': eval(parser.get('training', 'batch')),
            'epochs': eval(parser.get('training', 'epochs'))
        }

        for model in loaded_models:
            config = dict(ChainMap(config, model.read_config(parser)))

        configs.append(config)

    return configs


def export_metadata(config, df, pl):
    """
    Generation of metadata file for the models

    :param df: dataframe from dataset
    :param config: configuration on which the model is trained
    :param model_id: model id
    :param pl: saving_path
    :return: [model_name, id, data_source, targets, column_name]
    """

    print(type(config['model']))

    configparser = ConfigParser()
    configparser.add_section('model')
    configparser.set(section='model', option="name", value=config['model'])
    configparser.set(section='model', option="id", value=str(config['id']))

    configparser.add_section('data')
    configparser.set(section='data', option='columns', value=str(list(df.columns.values)))
    configparser.set(section='data', option='groups', value=str(config['groups']))
    configparser.set(section='data', option='targets', value=str(config['targets']))
    configparser.set(section='data', option='unknown-categoricals', value=str(config['uncats']))
    configparser.set(section='data', option='unknown-reels', value=str(config['unreels']))
    configparser.set(section='data', option='known-categoricals', value=str(config['kncats']))
    configparser.set(section='data', option='known-reels', value=str(config['knreels']))

    configparser.add_section('training')
    configparser.set(section='training', option='batch-size', value=str(config['batch']))

    with open(Path(pl) / f'metadata-{datetime.now().strftime("%H%M%S")}-{config["epochs"]}.cfg', 'w') as configfile:
        configparser.write(configfile)


def read_metadata(file):
    """
   Reads the metadata file that was generated when the model was trained

   :param file: location of the file
   :return: [model_name, model_id, data_source, targets, column_names]
   """
    configparser = ConfigParser()
    configparser.read(file)

    return {
        'model': configparser.get('model', 'name'),
        'id': configparser.get('model', 'id'),
        'columns': eval(configparser.get('data', 'columns')),
        'groups': eval(configparser.get('data', 'groups')),
        'targets': eval(configparser.get('data', 'targets')),
        'uncats': eval(configparser.get('data', 'unknown-categoricals')),
        'unreels': eval(configparser.get('data', 'unknown-reels')),
        'kncats': eval(configparser.get('data', 'known-categoricals')),
        'knreels': eval(configparser.get('data', 'known-reels')),
        'batch': int(configparser.get('training', 'batch-size'))
    }
