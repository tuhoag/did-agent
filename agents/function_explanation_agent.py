import json
from typing import Annotated, Optional, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
import operator

from agents.model_factory import Model, ModelFactory


class ArgumentInfo(TypedDict):
    name: str
    type: str

class FunctionInfo(TypedDict):
    name: str
    arguments: list[ArgumentInfo]
    return_type: str
    description: str

class FunctionExplanationAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    function_code: str
    explanation: FunctionInfo
    functions_explanation_result: str
    error: Optional[str]
    num_tries: int

def parse_explanation_json(json_str: str) -> tuple[FunctionInfo, str]:
    error = ""
    parsed: FunctionInfo = {}
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            error = "Output is not a JSON object."
    except json.JSONDecodeError as e:
        error = f"Invalid JSON: {str(e)}"

    return parsed, error

class FunctionExplanationAgent:
    def __init__(self, model: Model, max_tries: int = 3):
        self.model = model
        self.max_tries = max_tries

        self.extract_system = SystemMessage(
            content=(
                """You are a code analysis agent. Explain the given function code, preserving arguments and return types if present. 
                Return the result as a JSON object describing the function's name, arguments, return type, and a brief description.                 
                
                Schema:                
                {
                    name: "function name",
                    arguments: [
                        {name: "arg1", type: "type1"},
                        {name: "arg2", type: "type2"},
                    ],
                    return_type: "return type",
                    description: "brief description of the function"
                }                    
                
                Requirements:                
                - Preserve function signature, including argument names and types, and return types.
                - Return ONLY the function explanation in a JSON object.
                - No additional text outside the JSON object.
                """
            )
        )

        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(FunctionExplanationAgentState)

        builder.add_node("explain_function", self.explain)
        builder.add_node("validate_explanation", self.validate)

        builder.add_edge(START, "explain_function")
        builder.add_edge("explain_function", "validate_explanation")
        builder.add_conditional_edges(
            "validate_explanation",
            self.should_continue,
            {
                True: "explain_function",
                False: END
            }
        )
        return builder.compile()

    def explain(self, state: FunctionExplanationAgentState) -> FunctionExplanationAgentState:
        # print(f"Explain function from: {state['function_code'][:100]}...")
        # If file is empty, skip extraction
        if len(state["messages"]) == 0:
            state["messages"] = [self.extract_system, HumanMessage(content=state['function_code'])]

        num_tries = state.get("num_tries", 0) + 1

        if not state['function_code'].strip():
            return {**state, "functions_explanation_result": "{}", "num_tries": num_tries}

        response = self.model.invoke(state["messages"])
        # print(f"Explanation response: {response.content}")

        return {"functions_explanation_result": response.content, "messages": [AIMessage(content=response.content)], "num_tries": num_tries}

    def validate(self, state: FunctionExplanationAgentState) -> FunctionExplanationAgentState:
        # print("Validate explanation")
        explanation, error = parse_explanation_json(state['functions_explanation_result'])
        # print(f"Parsed functions: {functions_code}, error: {error}")

        if error != "":
            error_message = SystemMessage(content=error)
            return {**state, "error": error, "messages": [error_message]}

        return {**state, "explanation": explanation, "error": None}

    def should_continue(self, state: FunctionExplanationAgentState) -> bool:
        if state["num_tries"] >= self.max_tries:
            return False
        return state["error"] is not None

    def invoke(self, initial_state: FunctionExplanationAgentState) -> FunctionExplanationAgentState:
        return self.graph.invoke(initial_state)