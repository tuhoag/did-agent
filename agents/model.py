from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
# from vllm import LLM, SamplingParams
from langchain_community.llms import VLLM
import logging

logging.getLogger("vllm").setLevel(logging.ERROR)

class Model:
    name: str
    backend: str
    model: ChatOllama | VLLM

    def __init__(self, name: str, backend: str, temperature: float = 0.0):
        self.name = name
        self.backend = backend
        self.temperature = temperature

        if backend == "ollama":
            self.model = ChatOllama(model=name, temperature=temperature)
        elif backend == "vllm":
            # Initialize VLLM model here
            # self.model = VLLM(model=name)
            # sampling_params = SamplingParams(temperature=temperature, top_p=0.95)
            self.model = VLLM(model=name, temperature=temperature, disable_log_stats = True)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def invoke(self, messages) -> AIMessage:
        if self.backend == "ollama":
            return self.model.invoke(messages)
        elif self.backend == "vllm":
            response = self.model.invoke(messages)
            return AIMessage(content=response)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        # if self.backend == "ollama":
        #     return self.model.invoke(messages)
        # elif self.backend == "vllm":
        #     return self.model.invoke(messages)
        # else:
        #     raise ValueError(f"Unsupported backend: {self.backend}")