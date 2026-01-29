from typing import Optional
from langchain_ollama import ChatOllama

from agents.model import Model


class ModelFactory:
    name2model = {}

    def __init__(self):
        self.name2model = {}

    def register_model(self, model_name: str, model_backend: str) -> Optional[Model]:
        try:
            key = f"{model_name}"
            model = self.name2model.get(key, None)
            if model is None or model.backend != model_backend:
                model = Model(model_name, model_backend)
                self.name2model[key] = model
            return model
        except ValueError as e:
            raise e

    def get_model(self, model_name: str) -> Optional[Model]:
        key = f"{model_name}"
        return self.name2model.get(key, None)