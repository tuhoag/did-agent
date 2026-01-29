from langchain_ollama import ChatOllama

from agents.function_explanation_agent import FunctionExplanationAgent, FunctionExplanationAgentState
from agents.functions_extractor_agent import FunctionExtractorAgent, FunctionExtractorAgentState



class CodeExtractorAgent:
    def __init__(self, llm: ChatOllama, max_tries: int = 3):
        self.llm = llm
        self.max_tries = max_tries

        self.functions_extractor_agent = FunctionExtractorAgent(llm, max_tries=max_tries)
        self.function_explanation_agent = FunctionExplanationAgent(llm, max_tries=max_tries)

    def __call__(self, code: str) -> dict:
        # print(code)
        initial_state: FunctionExtractorAgentState = {
            'code': code,
            'functions': [],
            'functions_extraction_result': "",
            'error': None,
            'messages': [],
        }
        final_state = self.functions_extractor_agent.invoke(initial_state)

        codes = final_state["functions"]
        # print(f"Extracted {len(codes)} functions")
        detail_explanations = []
        for idx, code in enumerate(codes):
            # print(f"Function code {idx}:\n{code}\n---")

            explanation_state: FunctionExplanationAgentState = {
                'function_code': code,
                'functions_explanation_result': "",
                'explanation': {},
                'error': None,
                'messages': [],
                'num_tries': 0,
            }
            explanation_result = self.function_explanation_agent.invoke(explanation_state)

            # print(f"Explanation result for function {idx}: {explanation_result}")
            # raise Exception("Stop here for debugging")
            detail_explanations.append(explanation_result['explanation'])
            # print(f"Function explanation {idx}:\n{explanation_result['explanation']}\n===")

        # print(f"Final explanations: {detail_explanations}")

        return {"functions": detail_explanations}
