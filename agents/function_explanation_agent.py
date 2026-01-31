import json
from typing import Annotated, Optional, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
import operator

from agents.model_factory import Model, ModelFactory

class FunctionExplanationAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    function_code: str
    explanation: Annotated[list[str], operator.add]
    num_tries: int

class FunctionExplanationAgent:
    def __init__(self, model: Model, max_tries: int = 3):
        self.model = model
        self.max_tries = max_tries

        self.extract_system = SystemMessage(
            content=(
                """Your task is to explain the provided Rust code. You must explain parameters, return values, and overall functionality.             
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


        response = self.model.invoke(state["messages"])
        # print(f"Explanation response: {response.content}")

        return {"explanation": [response.content], "messages": [AIMessage(content=response.content)], "num_tries": num_tries}

    def validate(self, state: FunctionExplanationAgentState) -> FunctionExplanationAgentState:
        # print("Validate explanation")
        validate_system = SystemMessage(
            content=(
                f"""Evaluate the following LLM response to the given user question. Check if the explanation correctly describes the function's parameters, return values, and overall functionality.
                
                User question:
                {state["function_code"]}

                LLM response:
                {state["explanation"]}
                
                Assess the response on the following dimensions:
                    - Completeness: Does the explanation cover parameters, return values, and overall functionality?
                    - Clarity: Is the explanation clear and easy to understand?
                
                For each dimension:
                    - Give a score from 1 (poor) to 5 (excellent)
                    - Provide a brief justification
                    
                Then:
                    - List any incorrect or misleading statements, if present
                    - Suggest specific improvements
                    - Give an overall verdict: Excellent / Good / Fair / Poor
                """
            )
        )
        state["messages"].append(validate_system)
        response = self.model.invoke(state["messages"])

        return {
            "messages": [AIMessage(content=response.content)],
        }


    def should_continue(self, state: FunctionExplanationAgentState) -> bool:
        validatation_message = state["messages"][-1].content.lower()
        if "poor" in validatation_message and state["num_tries"] < self.max_tries:
            return True

        return False

    def invoke(self, initial_state: FunctionExplanationAgentState) -> FunctionExplanationAgentState:
        return self.graph.invoke(initial_state)