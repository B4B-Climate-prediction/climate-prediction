"""A pytorch implementation to train, load & predict a forecasting model"""

import os
import pickle
import shutil

import pandas as pd
from abc import ABC
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import glob

from model import Model
from utils import config_reader


class Tft(Model, ABC):
    """Temporal Fusion Transformer model"""

    name = "Tft"

    read_metadata = lambda configparser, **kwargs: read_metadata(configparser, **kwargs)
    generate_config = lambda configparser, **kwargs: generate_config(configparser, **kwargs)

    def __init__(self, metadata, data):
        super().__init__(metadata, data)
        self.main_config = config_reader.read_main_config()

    def generate_time_series_dataset(self):
        """
        Generates a TimeSeriesDataSet

        :return: TimeSeriesDataSet
        """
        targets = self.metadata['targets']

        if len(targets) == 1:
            target = targets[0]
        else:
            target = targets

        for k_cat in self.metadata['kncats']:
            self.data[k_cat] = str(self.data[k_cat])

        for uk_cat in self.metadata['uncats']:
            self.data[uk_cat] = str(self.data[uk_cat])

        return TimeSeriesDataSet(
            self.data,
            target=target,
            time_idx='Index',
            group_ids=self.metadata['groups'],
            min_encoder_length=self.metadata['min-encoder-length'],
            max_encoder_length=self.metadata['max-encoder-length'],
            min_prediction_length=self.metadata['min-prediction-length'],
            max_prediction_length=self.metadata['max-prediction-length'],
            time_varying_known_categoricals=self.metadata['kncats'],
            time_varying_known_reals=self.metadata['knreels'],
            time_varying_unknown_categoricals=self.metadata['uncats'],
            time_varying_unknown_reals=self.metadata['unreels'],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

    def generate_model(self, dataset):
        """
        Generates a temporal fusion transformer

        :return: TemporalFusionTransformer
        """

        targets = self.metadata['targets']

        if len(targets) > 1:
            output = [self.metadata['output-size'] for _ in targets]
        else:
            output = self.metadata['output-size']

        return TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=self.metadata['learning-rate'],
            hidden_size=self.metadata['hidden-size'],
            attention_head_size=self.metadata['attention-head-size'],
            dropout=self.metadata['dropout'],  # Check deze shit ook uit
            hidden_continuous_size=self.metadata['hidden-continuous-size'],
            output_size=output,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=self.metadata['reduce-on-plateau-patience']
        )

    def train_model(self, dataset, created_model, **kwargs):
        """
        Trains a TemporalFusionTransformer

        :return: A trainer object
        """
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode='min')

        train_dataloader, val_dataloader = self.create_data_loaders(dataset)

        logger = False
        if (self.main_config['wandb']) and (self.main_config['wandb-project'] is not None):
            logger = WandbLogger(project=self.main_config['wandb-project'])



        trainer = Trainer(
            max_epochs=kwargs['epochs'],
            gpus=self.metadata['gpus'],
            gradient_clip_val=self.metadata['gradient-clip-val'],
            limit_train_batches=self.metadata['limit-train-batches'],
            callbacks=[early_stop_callback],
            weights_save_path=str(Path(__file__).parent.parent / 'out' / 'models' / 'tft' / f'{self.model_id}'),
            default_root_dir='',
            logger=logger,
        )

        trainer.fit(
            created_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        # torch.save(created_model.state_dict(), os.path.join(Path(__file__).parent.parent / 'out' / 'models' / 'tft', 'weights.ckpt'))

        return trainer  # should return something that is obtainable by wandb!

    def predict(self, model_, **kwargs):
        """
        Predicts X amount of time-steps into the future.

        :param model_: trained model

        :return: predicted targets
        """

        predictions = [None for _ in range(self.metadata['max-encoder-length'])]

        unit = kwargs["timeunit"][1]  # E.g. 'minutes'
        value = int(kwargs['timeunit'][0])  # E.g. 10

        for j_lower in range(len(self.data)):
            j_upper = j_lower + self.metadata['max-encoder-length']

            if j_upper > (len(self.data) - 1):
                break

            encoder_data = self.data[j_lower:j_upper]

            last_data = encoder_data[lambda x: x.Index == x.Index.max()].copy()

            decoder_data = pd.concat(
                [last_data.assign(Timestamp=lambda y: y.Timestamp + pd.DateOffset(**{unit: value * i})) for i in
                 range(1, self.metadata['max-prediction-length'] + 1)],
                ignore_index=True,
            )

            # add time index consistent with "data"
            decoder_data["Index"] += encoder_data["Index"].max() + decoder_data.index + 1 - decoder_data["Index"].min()

            # combine encoder and decoder data
            new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

            predictions.append((model_.predict(new_prediction_data)[0][0]).item())

        return predictions

    def tune_hyper_parameter(self, dataset, **kwargs):  # Add clean up after the creation of the best trial!
        """
        Hyper-tunes the TemporalFusionTransformer based on Trials & Epochs

        :param dataset: TimeSeriesDataSet

        :return: Best TemporalFusionTransformer
        """
        train_dataloader, val_dataloader = self.create_data_loaders(dataset)

        study = optimize_hyperparameters(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_path='hyp_tuning',
            max_epochs=kwargs['epochs'],
            n_trials=kwargs['trials'],
            attention_head_size_range=(
                self.metadata['min-attention-head-size'], self.metadata['max-attention-head-size']),
            gradient_clip_val_range=(self.metadata['min-gradient-clip-val'], self.metadata['max-gradient-clip-val']),
            hidden_size_range=(self.metadata['min-hidden-size'], self.metadata['max-hidden-size']),
            dropout_range=(self.metadata['min-dropout'], self.metadata['max-dropout']),
            hidden_continuous_size_range=(
                self.metadata['min-hidden-continuous-size'], self.metadata['max-hidden-continuous-size']),
            learning_rate_range=(self.metadata['min-learning-rate'], self.metadata['max-learning-rate']),
            log_dir=''
        )

        self.metadata['gradient-clip-val'] = study.best_params.get('gradient_clip_val')
        self.metadata['hidden-size'] = study.best_params.get('hidden_size')
        self.metadata['dropout'] = study.best_params.get('dropout')
        self.metadata['hidden-continuous-size'] = study.best_params.get('hidden_continuous_size')
        self.metadata['attention-head-size'] = study.best_params.get('attention_head_size')
        self.metadata['learning-rate'] = study.best_params.get('learning_rate')

        best_model_path = 'hyp_tuning' + "/trial_" + str(study.best_trial.number)
        list_of_files = glob.glob(best_model_path)  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        dst_path = str(
            Path(__file__).parent.parent / 'out' / 'models' / 'tft' / str(self.metadata['id']) / 'checkpoints')
        shutil.move(latest_file, dst_path)
        shutil.rmtree('hyp_tuning')

        weights_file = os.listdir(dst_path)[0]

        # SHOULD RETURN THE BEST TRIAL! IF THIS FUNCTION IS AVAILABLE!
        return TemporalFusionTransformer.load_from_checkpoint(f"{dst_path}/{weights_file}")

    def evaluate_model(self, evaluated_model, dataset):
        """
        Evaluates the model based on performance.

        :param dataset: TimeSeriesDataSet
        :param evaluated_model: TemporalFusionTransformer

        :return: Nothing
        """
        _, validation_data_loader = self.create_data_loaders(dataset)

        raw_predictions, x = evaluated_model.predict(validation_data_loader, mode="raw", return_x=True)

        for i in range(len(x)):
            evaluated_model.plot_prediction(x, raw_predictions, idx=i, add_loss_to_title=True)

    def create_data_loaders(self, dataset):
        """
        For the TemporalFusionTransformer it requires data-loaders
        This function is to split the dataset into a validation & training set.

        :param dataset: TimeSeriesDataSet

        :return: DataLoaders
        """
        validation = TimeSeriesDataSet.from_dataset(dataset, self.data, predict=True, stop_randomization=True)

        return dataset.to_dataloader(train=True, batch_size=self.metadata['batch'], num_workers=2,
                                     shuffle=False), validation.to_dataloader(train=False,
                                                                              batch_size=self.metadata['batch'],
                                                                              num_workers=2, shuffle=False)

    def load_model(self, path, **kwargs):
        """
        Load in TemporalFusionTransformer

        :return: TemporalFusionTransformer
        """
        return TemporalFusionTransformer.load_from_checkpoint(path)

    def write_metadata(self, configparser):
        """
        Writes the metadata from self.metadata into the configparser

        :param configparser: The configparser that writes to the config file for the model

        :return: nothing
        """
        configparser.set(section='training', option='min-encoder-length',
                         value=str(self.metadata['min-encoder-length']))
        configparser.set(section='training', option='max-encoder-length',
                         value=str(self.metadata['max-encoder-length']))
        configparser.set(section='training', option='min-prediction-length',
                         value=str(self.metadata['min-prediction-length']))
        configparser.set(section='training', option='max-prediction-length',
                         value=str(self.metadata['max-prediction-length']))
        configparser.set(section='training', option='hidden-size', value=str(self.metadata['hidden-size']))
        configparser.set(section='training', option='dropout', value=str(self.metadata['dropout']))
        configparser.set(section='training', option='attention-head-size',
                         value=str(self.metadata['attention-head-size']))
        configparser.set(section='training', option='hidden-continuous-size',
                         value=str(self.metadata['hidden-continuous-size']))
        configparser.set(section='training', option='output-size', value=str(self.metadata['output-size']))
        configparser.set(section='training', option='reduce-on-plateau-patience',
                         value=str(self.metadata['reduce-on-plateau-patience']))
        configparser.set(section='training', option='gradient-clip-val', value=str(self.metadata['gradient-clip-val']))
        configparser.set(section='training', option='gpus', value=str(self.metadata['gpus']))
        configparser.set(section='training', option='limit-train-batches',
                         value=str(self.metadata['limit-train-batches']))


def read_metadata(configparser, **kwargs):
    """
    Reads the metadata from a specific config file

    :param configparser: The configparser that reads the file
    :param kwargs: Contains one argument if the model is being hyper-tuned

    :return: dictonary of settings
    """
    if configparser.get('model', 'name') == 'Tft':
        settings = {'min-encoder-length': eval(configparser.get('training', 'min-encoder-length')),
                    'max-encoder-length': eval(configparser.get('training', 'max-encoder-length')),
                    'min-prediction-length': eval(configparser.get('training', 'min-prediction-length')),
                    'max-prediction-length': eval(configparser.get('training', 'max-prediction-length')),
                    'reduce-on-plateau-patience': eval(configparser.get('training', 'reduce-on-plateau-patience')),
                    'gpus': eval(configparser.get('training', 'gpus')),
                    'limit-train-batches': eval(configparser.get('training', 'limit-train-batches')),
                    'output-size': eval(configparser.get('training', 'output-size'))
                    }

        if not kwargs['hyper-tuning']:
            settings['hidden-size'] = eval(configparser.get('training', 'hidden-size'))
            settings['dropout'] = eval(configparser.get('training', 'dropout'))
            settings['attention-head-size'] = eval(configparser.get('training', 'attention-head-size'))
            settings['hidden-continuous-size'] = eval(configparser.get('training', 'hidden-continuous-size'))
            settings['gradient-clip-val'] = eval(configparser.get('training', 'gradient-clip-val'))

        else:
            settings['min-attention-head-size'] = eval(configparser.get('hyper-tuning', 'min-attention-head-size'))
            settings['max-attention-head-size'] = eval(configparser.get('hyper-tuning', 'max-attention-head-size'))

            settings['min-gradient-clip-val'] = eval(configparser.get('hyper-tuning', 'min-gradient-clip-val'))
            settings['max-gradient-clip-val'] = eval(configparser.get('hyper-tuning', 'max-gradient-clip-val'))

            settings['min-hidden-size'] = eval(configparser.get('hyper-tuning', 'min-hidden-size'))
            settings['max-hidden-size'] = eval(configparser.get('hyper-tuning', 'max-hidden-size'))

            settings['min-dropout'] = eval(configparser.get('hyper-tuning', 'min-dropout'))
            settings['max-dropout'] = eval(configparser.get('hyper-tuning', 'max-dropout'))

            settings['min-hidden-continuous-size'] = eval(
                configparser.get('hyper-tuning', 'min-hidden-continuous-size'))
            settings['max-hidden-continuous-size'] = eval(
                configparser.get('hyper-tuning', 'max-hidden-continuous-size'))

            settings['min-learning-rate'] = eval(configparser.get('hyper-tuning', 'min-learning-rate'))
            settings['max-learning-rate'] = eval(configparser.get('hyper-tuning', 'max-learning-rate'))

        return settings


def generate_config(configparser, **kwargs):
    """
    Generates a example model-config for the generate_model.py

    :param configparser: The configparser that writes to the file
    :param kwargs: Contains if the model is being hyper-tuned

    :return: dictionary of settings
    """
    configparser.set(section='training', option='min-encoder-length', value='0')
    configparser.set(section='training', option='max-encoder-length', value='27')
    configparser.set(section='training', option='min-prediction-length', value='1')
    configparser.set(section='training', option='max-prediction-length', value='1')
    configparser.set(section='training', option='output-size', value='7')
    configparser.set(section='training', option='reduce-on-plateau-patience', value='4')
    configparser.set(section='training', option='gpus', value='0')
    configparser.set(section='training', option='limit-train-batches', value='50')

    if not kwargs['hyper']:
        configparser.set(section='training', option='hidden-size', value='16')
        configparser.set(section='training', option='dropout', value='0.1')
        configparser.set(section='training', option='attention-head-size', value='1')
        configparser.set(section='training', option='hidden-continuous-size', value='8')
        configparser.set(section='training', option='gradient-clip-val', value='0.15')

    else:
        configparser.add_section('hyper-tuning')
        configparser.set(section='hyper-tuning', option='min-attention-head-size', value='1')
        configparser.set(section='hyper-tuning', option='max-attention-head-size', value='4')

        configparser.set(section='hyper-tuning', option='min-gradient-clip-val', value='0.01')
        configparser.set(section='hyper-tuning', option='max-gradient-clip-val', value='100')

        configparser.set(section='hyper-tuning', option='min-hidden-size', value='16')
        configparser.set(section='hyper-tuning', option='max-hidden-size', value='265')

        configparser.set(section='hyper-tuning', option='min-dropout', value='0.1')
        configparser.set(section='hyper-tuning', option='max-dropout', value='0.3')

        configparser.set(section='hyper-tuning', option='min-hidden-continuous-size', value='8')
        configparser.set(section='hyper-tuning', option='max-hidden-continuous-size', value='64')

        configparser.set(section='hyper-tuning', option='min-learning-rate', value='1e-5')
        configparser.set(section='hyper-tuning', option='max-learning-rate', value='1.0')
