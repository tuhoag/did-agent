import glob
import json
from os import path

from langchain_ollama import ChatOllama
from tqdm import tqdm

from agents.code_extractor_agent import CodeExtractorAgent

OUTPUT_DIR = "./output"
DATA_DIR = "./data"
model_name = "qwen2.5"
model = ChatOllama(model=model_name, temperature=0)

def get_output_path(lib_name: str) -> str:
    return path.join(OUTPUT_DIR, f"{lib_name}_kg.json")

def extract_library_metadata(library_path: str) -> dict:
    metadata = {}

    # print(f"Extracting metadata from library at {library_path}")
    # print(path.basename(library_path))
    splits = path.basename(library_path).split("-")
    metadata["name"] = splits[0]
    metadata["version"] = "-".join(splits[1:])

    # print(f"Extracted metadata for library {splits[0]}: {metadata}")
    return metadata

def extract_functions(file: str) -> dict:
    # read file and check how many functions are there in the file
    agent = CodeExtractorAgent(llm=model)
    with open(file, "r") as f:
        code = f.read()
        functions = agent(code)
        # print(f"Extracted {len(functions['functions'])} functions from file {file}")
        # print(functions)

        # raise Exception("Stop here for debugging")
        return functions

def extract_file_path_metadata(file_path: str) -> dict:
    metadata = {'path': file_path}
    # extract the path components
    components = file_path.split(path.sep)
    # find the index of 'src'
    if 'src' in components:
        src_index = components.index('src')
        relative_path_components = components[src_index + 1:-1]  # exclude 'src' and the file name
        metadata['module_path'] = ".".join(relative_path_components)
    else:
        metadata['module_path'] = ""
    return metadata

def extract_library_documentation(library_path: str) -> list:
    # read all .rs files in all subdirectories
    rs_files = glob.glob(path.join(library_path, "**", "*.rs"), recursive=True)
    # print(f"Found {len(rs_files)} .rs files in library at {library_path}")

    data = []
    for file in tqdm(rs_files):
        file_info = {}
        file_metadata = extract_file_path_metadata(file)
        extracted_functions = extract_functions(file)

        file_info.update(file_metadata)
        # print(f"Extracted functions from file {file}: {extracted_functions}")
        file_info.update(extracted_functions)

        data.append(file_info)

        # print(file_info)

    return data

def find_all_libraries():
    libraries = glob.glob(path.join(DATA_DIR, "*"))
    # libraries = ['./data/sample-1.1.0']
    # print(f"Found {len(libraries)} libraries:")

    for lib in tqdm(libraries):
        # print(f"- {lib}")
        lib_info = {}

        metadata_info = extract_library_metadata(lib)
        code_info = extract_library_documentation(lib)

        lib_info.update(metadata_info)
        lib_info.update({"code": code_info})

        # all_libs_info.append(lib_info)
        # print(f"Library info: {json.dumps(lib_info, indent=2)}")
        json.dump(lib_info, open(get_output_path("-".join([metadata_info["name"], metadata_info["version"]])), "w"), indent=2)

def main():
    print("Hello from kg-construction!")

    find_all_libraries()


if __name__ == "__main__":
    main()