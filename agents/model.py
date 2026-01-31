from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
# from vllm import LLM, SamplingParams
from langchain_community.llms import VLLM
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

class Model:
    name: str
    backend: str
    model: ChatOllama | VLLM

    def __init__(self, name: str, backend: str, temperature: float = 0.0):
        self.name = name
        self.backend = backend
        self.temperature = temperature

        if backend == "ollama":
            self.model = ChatOllama(
                model=name,
                temperature=temperature,

            )
        elif backend == "vllm":
            # Initialize VLLM model here
            # self.model = VLLM(model=name)
            # sampling_params = SamplingParams(temperature=temperature, top_p=0.95)
            # self.model = VLLM(model=name, temperature=temperature, base_url="http://localhost:8000")
            self.model = ChatOpenAI(
                model_name=name,
                temperature=temperature,
                openai_api_base="http://localhost:8000/v1",
                api_key=""
            )
            # self.model = init_chat_model(
            #     model=name,
            #     model_provider="openai",
            #     # model_provider="http://localhost:8000",
            #     base_url="http://localhost:8000/v1",
            #     api_key="",

            # )
            # self.model = ChatOpenAI(model_name=name, temperature=temperature, openai_api_base="http://127.0.0.1:8000")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def invoke(self, messages) -> AIMessage:
        # print(f"Invoking model {self.name} with backend {self.backend}")
        try:
            if self.backend == "ollama":
                return self.model.invoke(messages)
            elif self.backend == "vllm":
                response = self.model.invoke(messages)
                # response is already an AIMessage, return it directly
                if isinstance(response, AIMessage):
                    return response
                else:
                    # If it's a string, wrap it in AIMessage
                    return AIMessage(content=response.content if hasattr(response, 'content') else str(response))
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except Exception as e:
            # print(f"Error invoking model {self.name} with backend {self.backend}: {str(e)}")
            raise e
        # if self.backend == "ollama":
        #     return self.model.invoke(messages)
        # elif self.backend == "vllm":
        #     return self.model.invoke(messages)
        # else:
        #     raise ValueError(f"Unsupported backend: {self.backend}")