import json
from typing import Annotated, Optional, TypedDict
from agents.model_factory import Model, ModelFactory
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
import operator

class FunctionExtractorAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    # agent 1
    code: str
    functions: list[str]
    functions_extraction_result: str
    error: Optional[str]
    num_tries: int



def parse_extraction_json(json_str: str) -> tuple[list, str]:
    error = ""
    parsed = []
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, list):
            error = "Output is not a JSON array."
    except json.JSONDecodeError as e:
        error = f"Invalid JSON: {str(e)}"

    return parsed, error

class FunctionExtractorAgent:
    def __init__(self, model: Model, max_tries: int = 3):
        self.model = model
        self.max_tries = max_tries

        self.extract_system = SystemMessage(
            content=(
                """You are a code analysis agent. Extract ALL function definitions 
                from the given code, preserving arguments and return types if present. 
                Return the result as a JSON array of strings, where each string contains one function's code.                 
                
                Schema:                
                [
                    "function code 1",
                    "function code 2",
                ]
                
                Requirements:
                - Return ONLY the function code strings in a JSON array.
                - Preserve function signatures, including argument names and types, and return types.
                """
            )
        )

        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(FunctionExtractorAgentState)

        builder.add_node("extract_functions", self.extract)
        builder.add_node("validate_extraction", self.validate)

        builder.add_edge(START, "extract_functions")
        builder.add_edge("extract_functions", "validate_extraction")
        builder.add_conditional_edges(
            "validate_extraction",
            self.should_continue,
            {
                True: "extract_functions",
                False: END
            }
        )
        return builder.compile()

    def extract(self, state: FunctionExtractorAgentState) -> FunctionExtractorAgentState:
        # print(f"Extract functions from: {state['code'][:100]}...")
        # If file is empty, skip extraction
        if len(state["messages"]) == 0:
            state["messages"] = [self.extract_system, HumanMessage(content=state['code'])]

        num_tries = state.get("num_tries", 0) + 1

        if not state['code'].strip():
            return {**state, "functions_extraction_result": "[]", "num_tries": num_tries}

        response = self.model.invoke(state["messages"])
        print(f"Extraction response: {response}")

        return {"functions_extraction_result": response.content, "messages": [AIMessage(content=response.content)], "num_tries": num_tries}

    def validate(self, state: FunctionExtractorAgentState) -> FunctionExtractorAgentState:
        print("Validate extraction")
        functions_code, error = parse_extraction_json(state['functions_extraction_result'])
        # print(f"Parsed functions: {functions_code}, error: {error}")

        if error != "":
            error_message = SystemMessage(content=error)
            return {**state, "error": error, "messages": [error_message]}

        return {**state, "functions": functions_code, "error": None}

    def should_continue(self, state: FunctionExtractorAgentState) -> bool:
        if state["num_tries"] >= self.max_tries:
            return False
        return state["error"] is not None

    def invoke(self, initial_state: FunctionExtractorAgentState) -> FunctionExtractorAgentState:
        return self.graph.invoke(initial_state)