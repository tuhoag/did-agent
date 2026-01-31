import os
from typing import TypedDict
from tree_sitter import Language, Parser
import tree_sitter_rust as rpython

from agents.function_explanation_agent import FunctionExplanationAgent, FunctionExplanationAgentState
from agents.model_factory import Model


class FunctionInfo(TypedDict):
    name: str
    signature: str
    code: str
    description: str

class CodeFileInfo(TypedDict):
    functions: list[FunctionInfo]
    language: str

class FunctionExtractor():
    function_extractor = {}

    def __init__(self):
        self.function_extractor = {
            "rust": self.extract_rust_functions,
        }

    def is_supported(self, language: str) -> bool:
        return language in self.function_extractor

    def __call__(self, file: str) -> CodeFileInfo:
        # detect programming language
        language = detect_programming_language(file)
        if self.is_supported(language) == False:
            return {"language": language, "functions": []}

        with open(file, "r") as f:
            code = f.read()
            functions = self.function_extractor[language](code)

        return {
            "language": language,
            "functions": functions
        }

    def extract_rust_functions(self, code: str) -> list[FunctionInfo]:
        # use tree-sitter to parse Rust code and extract function definitions
        R_LANGUAGE = Language(rpython.language())
        parser = Parser(R_LANGUAGE)

        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        functions = []

        for child in root_node.children:
            if child.type == "function_item":
                name = None
                visibility = None
                start_signature = end_signature = child.start_byte

                for grandchild in child.children:
                    if grandchild.type == "fn":
                        start_signature = grandchild.start_byte
                    if grandchild.type == "block":
                        end_signature = grandchild.start_byte
                    if grandchild.type == "identifier":
                        name = code[grandchild.start_byte:grandchild.end_byte]
                    if grandchild.type == "visibility_modifier":
                        visibility = code[grandchild.start_byte:grandchild.end_byte]

                if name == "main" or visibility != "pub":
                    continue  # skip non-public functions and main function

                functions.append({
                    "name": name,
                    "signature": code[start_signature:end_signature - 1],
                    "code": code[child.start_byte:child.end_byte],
                })

        return functions


def detect_programming_language(file: str) -> str:
    # very naive detection based on keywords
    base_name = os.path.basename(file)
    if base_name.endswith(".rs"):
        return "rust"
    elif base_name.endswith(".py"):
        return "python"
    else:
        return "unknown"

class CodeExtractorAgent:
    def __init__(self, model: Model, max_tries: int = 3):
        self.model = model
        self.max_tries = max_tries

        self.function_explanation_agent = FunctionExplanationAgent(model, max_tries=max_tries)
    def __call__(self, file: str) -> CodeFileInfo:
        function_extractor = FunctionExtractor()
        functions = function_extractor(file)

        print(f"Extracted {len(functions['functions'])} functions from file {file}")
        # for idx, func in enumerate(functions['functions']):
        #     print(f"Function {idx}:\n{func}\n---")

        # raise Exception("Stop here for debugging")
        # print(f"Extracted {len(codes)} functions")

        for idx, func_info in enumerate(functions['functions']):
            # print(f"Function code {idx}:\n{code}\n---")

            explanation_state: FunctionExplanationAgentState = {
                'function_code': func_info['code'],
                'explanation': [],
                'messages': [],
            }
            explanation_result = self.function_explanation_agent.invoke(explanation_state)

            func_info['description'] = explanation_result['explanation'][-1]
            # print(f"Function code {idx}:\n{func_info['code']}\n---")
            # # print(f"{explanation_result['explanation']}")
            # print("--------------------------------")
            # for explanation in explanation_result['explanation']:
            #     print(f"Explanation part:\n{explanation}\n---")

            # print("================================")
            # for msg in explanation_result['messages']:
            #     print(f"Message: {msg}\n---")
            # raise Exception("Stop here for debugging")

            # print(f"Function explanation {idx}:\n{explanation_result['explanation']}\n===")

        # print(f"Final explanations: {detail_explanations}")

        return functions
