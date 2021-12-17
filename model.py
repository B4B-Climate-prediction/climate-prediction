from typing import Any, List


class Model:

    # Check if it possible to ArgsParser to each class
    # TODO: Check which parameters we want to add to this

    def __init__(self, data):
        self.data = data

    def generate_time_series_dataset(self, data, **kwargs) -> Any:
        raise NotImplementedError()

    def generate_model(self, data, **kwargs) -> Any:
        raise NotImplementedError()

    def train_model(self, data, model, **kwargs) -> Any:
        raise NotImplementedError()

    def load_model(self, **kwargs) -> Any:
        raise NotImplementedError

    def predict(self, **kwargs) -> List:
        raise NotImplementedError()

    def hyper_parameter_tuning(self, **kwargs):
        raise NotImplementedError()

