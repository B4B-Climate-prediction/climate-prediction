"""A pytorch implementation to train, load & predict a forecasting model"""

import os
import pickle
import pandas as pd
from abc import ABC
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model import Model


class Tft(Model, ABC):
    """Temporal Fusion Transformer model"""

    name = "Tft"

    read_metadata = lambda configparser: read_metadata(configparser)

    generate_config = lambda configparser: generate_config(configparser)

    def __init__(self, model_id, metadata, data):
        super().__init__(model_id, metadata, data)
        self.max_prediction_length = 6
        self.max_encoder_length = 24

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
            min_encoder_length=1,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
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
            output = [7 for _ in targets]
        else:
            output = 7

        return TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.01,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,  # Check deze shit ook uit
            hidden_continuous_size=8,
            output_size=output,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4
        )

    def train_model(self, dataset, created_model, **kwargs):
        """
        Trains a TemporalFusionTransformer

        :return: A trainer object
        """
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode='min')

        train_dataloader, val_dataloader = self.create_data_loaders(dataset)

        logger = False

        if kwargs['wandb'] is not None and kwargs['wandbproject'] is not None:
            logger = WandbLogger(project=kwargs['wandbproject'])

        trainer = Trainer(
            max_epochs=kwargs['epochs'],
            gpus=0,
            gradient_clip_val=0.15,
            limit_train_batches=50,
            callbacks=[early_stop_callback],
            weights_save_path=str(Path(__file__).parent.parent / 'out' / 'models' / 'tft' / f'{self.model_id}'),
            default_root_dir='',
            logger=logger
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

        predictions = [None for _ in range(self.max_encoder_length)]

        for j_lower in range(len(self.data)):
            j_upper = j_lower + self.max_encoder_length

            if j_upper > (len(self.data) - 1):
                break

            encoder_data = self.data[j_lower:j_upper]

            last_data = encoder_data[lambda x: x.Index == x.Index.max()].copy()

            for column in last_data.columns:
                if column in self.metadata['targets']:
                    continue
                if (column == 'Timestamp') or (column == 'Index'):
                    continue

                last_data.loc[last_data.Index.max(), column] = None

            unit = kwargs["timeunit"][1]  # E.g. 'minutes'
            value = int(kwargs['timeunit'][0])  # E.g. 10

            decoder_data = pd.concat(
                [last_data.assign(Timestamp=lambda y: y.Timestamp + pd.DateOffset(**{unit: value * i})) for i in
                 range(1, self.max_prediction_length + 1)],
                ignore_index=True,
            )

            # add time index consistent with "data"
            decoder_data["Index"] += encoder_data["Index"].max() + decoder_data.index + 1 - decoder_data["Index"].min()

            # combine encoder and decoder data
            new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

            predictions.append((model_.predict(new_prediction_data)[0][0]).item())

        return predictions
        # temp = model_.predict(self.data)
        # print()
        #
        # return None

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
            model_path=Path(__file__) / kwargs['model'],
            max_epochs=kwargs['epochs'],
            n_trials=kwargs['trials']
        )

        with open('optimization_summary.pkl', 'wb') as fout:
            pickle.dump(study, fout)

        # Use PATHLIB! Todo: FIX!
        path = kwargs['model'] + "/trial_" + str(study.best_trial.number)

        files = os.listdir(path)

        # SHOULD RETURN THE BEST TRIAL! IF THIS FUNCTION IS AVAILABLE!
        return TemporalFusionTransformer.load_from_checkpoint(path + "/" + files[len(files) - 1])

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
                                     shuffle=False), validation.to_dataloader(train=False, batch_size=self.metadata['batch'],
                                                                              num_workers=2, shuffle=False)

    def load_model(self, path, **kwargs):
        """
        Load in TemporalFusionTransformer

        :return: TemporalFusionTransformer
        """
        return TemporalFusionTransformer.load_from_checkpoint(path)

    def write_metadata(self, configparser):
        configparser.set(section='training', option='encoder-length', value=str(self.metadata['encoder-length']))


def read_metadata(configparser):
    if configparser.has_option('training', 'encoder-length'):
        return {
            'encoder-length': eval(configparser.get('training', 'encoder-length'))
        }


def generate_config(configparser):
    configparser.set(section='training', option='encoder-length', value='0')
