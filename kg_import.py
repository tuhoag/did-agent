import json
import os
from pathlib import Path
from neo4j import GraphDatabase
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer


def setup_milvus_collection(collection_name: str = "function_embeddings"):
    """
    Setup Milvus collection for function embeddings.
    
    Args:
        collection_name: Name of the collection to create
    """
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="project_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="project_version", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # all-MiniLM-L6-v2 dimension
    ]

    schema = CollectionSchema(fields, description="Function embeddings for semantic search")
    collection = Collection(collection_name, schema)

    # Create index for vector field
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    print(f"✓ Created Milvus collection: {collection_name}")
    return collection


def import_to_milvus(output_folder: str, collection_name: str = "function_embeddings"):
    """
    Import function embeddings to Milvus.
    
    Args:
        output_folder: Path to folder containing KG JSON files
        collection_name: Name of the Milvus collection
    """
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Setup collection
    collection = setup_milvus_collection(collection_name)

    # Get all JSON files
    output_path = Path(output_folder)
    json_files = list(output_path.glob("*.json"))

    all_functions = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            kg_data = json.load(f)

        project_name = kg_data["name"]
        project_version = kg_data["version"]

        for code_file in kg_data.get("code", []):
            for func in code_file.get("functions", []):
                # Create text to embed (combine function info)
                text_to_embed = f"{func['name']}\n{func['signature']}\n{func.get('description', '')}"

                all_functions.append({
                    "function_name": func["name"],
                    "file_path": code_file["path"],
                    "project_name": project_name,
                    "project_version": project_version,
                    "text": text_to_embed,
                })

    if not all_functions:
        print("No functions found to import to Milvus")
        return

    print(f"Generating embeddings for {len(all_functions)} functions...")

    # Generate embeddings
    texts = [f["text"] for f in all_functions]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Prepare data for insertion
    data = [
        [f["function_name"] for f in all_functions],
        [f["file_path"] for f in all_functions],
        [f["project_name"] for f in all_functions],
        [f["project_version"] for f in all_functions],
        embeddings.tolist(),
    ]

    # Insert data
    collection.insert(data)
    collection.flush()

    # Load collection for searching
    collection.load()

    print(f"✓ Imported {len(all_functions)} function embeddings to Milvus")


def import_kg_files_to_neo4j(output_folder: str, uri: str, auth: tuple):
    """
    Read JSON files from output folder and import them to Neo4j.
    
    Args:
        output_folder: Path to folder containing KG JSON files
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
        auth: Tuple of (username, password) for Neo4j authentication
    """
    driver = GraphDatabase.driver(uri, auth=auth)

    try:
        # Get all JSON files from output folder
        output_path = Path(output_folder)
        json_files = list(output_path.glob("*.json"))

        print(f"Found {len(json_files)} JSON files to import")

        for json_file in json_files:
            print(f"\nImporting {json_file.name}...")

            with open(json_file, 'r') as f:
                kg_data = json.load(f)

            with driver.session() as session:
                # Import project node
                session.run("""
                    MERGE (p:Project {name: $name, version: $version})
                """, name=kg_data["name"], version=kg_data["version"])

                # Import code files and functions
                for code_file in kg_data.get("code", []):
                    # Create file node
                    session.run("""
                        MERGE (f:File {path: $path})
                        SET f.module_path = $module_path,
                            f.language = $language
                        WITH f
                        MATCH (p:Project {name: $project_name, version: $project_version})
                        MERGE (p)-[:HAS_FILE]->(f)
                    """,
                    path=code_file["path"],
                    module_path=code_file.get("module_path", ""),
                    language=code_file.get("language", ""),
                    project_name=kg_data["name"],
                    project_version=kg_data["version"])

                    # Import functions
                    for func in code_file.get("functions", []):
                        session.run("""
                            MERGE (fn:Function {name: $name, signature: $signature})
                            SET fn.code = $code,
                                fn.description = $description
                            WITH fn
                            MATCH (f:File {path: $file_path})
                            MERGE (f)-[:CONTAINS]->(fn)
                        """,
                        name=func["name"],
                        signature=func["signature"],
                        code=func["code"],
                        description=func.get("description", ""),
                        file_path=code_file["path"])

                print(f"✓ Imported {json_file.name}")

        print(f"\n✓ Successfully imported {len(json_files)} files to Neo4j")

    finally:
        driver.close()


def query_and_visualize(uri: str, auth: tuple):
    """
    Query Neo4j and display the graph structure.
    
    Args:
        uri: Neo4j connection URI
        auth: Tuple of (username, password)
    """
    driver = GraphDatabase.driver(uri, auth=auth)

    try:
        with driver.session() as session:
            # Query 1: Get all projects
            print("\n" + "="*60)
            print("PROJECTS IN DATABASE")
            print("="*60)
            result = session.run("MATCH (p:Project) RETURN p.name as name, p.version as version")
            for record in result:
                print(f"  • {record['name']} v{record['version']}")

            # Query 2: Get project structure
            print("\n" + "="*60)
            print("PROJECT STRUCTURE")
            print("="*60)
            result = session.run("""
                MATCH (p:Project)-[:HAS_FILE]->(f:File)-[:CONTAINS]->(fn:Function)
                RETURN p.name as project, p.version as version, 
                       f.path as file, count(fn) as function_count
                ORDER BY p.name, f.path
            """)
            current_project = None
            for record in result:
                project_label = f"{record['project']} v{record['version']}"
                if current_project != project_label:
                    print(f"\n📦 {project_label}")
                    current_project = project_label
                print(f"  📄 {record['file']} ({record['function_count']} functions)")

            # Query 3: Get all functions with signatures
            print("\n" + "="*60)
            print("FUNCTIONS")
            print("="*60)
            result = session.run("""
                MATCH (f:File)-[:CONTAINS]->(fn:Function)
                RETURN f.path as file, fn.name as name, fn.signature as signature
                ORDER BY f.path, fn.name
            """)
            current_file = None
            for record in result:
                if current_file != record['file']:
                    print(f"\n📄 {record['file']}")
                    current_file = record['file']
                print(f"  ⚡ {record['name']}")
                print(f"     {record['signature']}")

            # Query 4: Graph statistics
            print("\n" + "="*60)
            print("GRAPH STATISTICS")
            print("="*60)
            stats = session.run("""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_FILE]->(f:File)
                OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
                RETURN count(DISTINCT p) as projects,
                       count(DISTINCT f) as files,
                       count(DISTINCT fn) as functions
            """).single()
            print(f"  Projects:  {stats['projects']}")
            print(f"  Files:     {stats['files']}")
            print(f"  Functions: {stats['functions']}")

            # Query 5: Cypher query for Neo4j Browser visualization
            print("\n" + "="*60)
            print("NEO4J BROWSER QUERY (copy to visualize)")
            print("="*60)
            print("""
MATCH (p:Project)-[:HAS_FILE]->(f:File)-[:CONTAINS]->(fn:Function)
RETURN p, f, fn
LIMIT 100
            """)

    finally:
        driver.close()


def test_retrieval(output_folder: str, neo4j_uri: str, neo4j_auth: tuple,
                   collection_name: str = "function_embeddings", top_k: int = 5):
    """
    Test if all functions in JSON files can be retrieved through Milvus and Neo4j.
    
    Args:
        output_folder: Path to folder containing KG JSON files
        neo4j_uri: Neo4j connection URI
        neo4j_auth: Neo4j authentication tuple
        collection_name: Milvus collection name
        top_k: Number of top results to retrieve from Milvus
    """
    print("\n" + "="*60)
    print("TESTING RETRIEVAL FROM MILVUS + NEO4J")
    print("="*60)

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()

    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)

    # Load all functions from JSON files
    output_path = Path(output_folder)
    json_files = list(output_path.glob("*.json"))

    all_test_functions = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            kg_data = json.load(f)

        project_name = kg_data["name"]
        project_version = kg_data["version"]

        for code_file in kg_data.get("code", []):
            for func in code_file.get("functions", []):
                all_test_functions.append({
                    "function_name": func["name"],
                    "signature": func["signature"],
                    "file_path": code_file["path"],
                    "project_name": project_name,
                    "project_version": project_version,
                    "description": func.get("description", ""),
                    "code": func["code"],
                })

    print(f"Testing retrieval for {len(all_test_functions)} functions...\n")

    successful_retrievals = 0
    failed_retrievals = []

    try:
        for i, test_func in enumerate(all_test_functions):
            # Create query text (use description or function name)
            query_text = test_func["description"][:500] if test_func["description"] else test_func["function_name"]

            # Generate query embedding
            query_embedding = model.encode([query_text])[0]

            # Search in Milvus
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["function_name", "file_path", "project_name", "project_version"]
            )

            # Check if the function is in top results
            found = False
            matched_result = None

            for hit in results[0]:
                if (hit.entity.get("function_name") == test_func["function_name"] and
                    hit.entity.get("file_path") == test_func["file_path"] and
                    hit.entity.get("project_name") == test_func["project_name"]):
                    found = True
                    matched_result = hit
                    break

            if found:
                # Retrieve detailed information from Neo4j
                with neo4j_driver.session() as session:
                    result = session.run("""
                        MATCH (p:Project {name: $project_name, version: $project_version})
                              -[:HAS_FILE]->(f:File {path: $file_path})
                              -[:CONTAINS]->(fn:Function {name: $function_name})
                        RETURN fn.name as name, fn.signature as signature, 
                               fn.code as code, fn.description as description,
                               f.path as file_path, p.name as project_name, p.version as project_version
                    """,
                    project_name=test_func["project_name"],
                    project_version=test_func["project_version"],
                    file_path=test_func["file_path"],
                    function_name=test_func["function_name"]
                    ).single()

                    if result:
                        successful_retrievals += 1
                        print(f"[{i+1}/{len(all_test_functions)}] ✓ {test_func['function_name']}")
                        print(f"    Milvus Score: {matched_result.distance:.4f}")
                        print(f"    Neo4j: Found in {result['file_path']}")
                        print(f"    Signature: {result['signature'][:60]}...")
                    else:
                        failed_retrievals.append({
                            "function": test_func["function_name"],
                            "reason": "Found in Milvus but not in Neo4j"
                        })
                        print(f"[{i+1}/{len(all_test_functions)}] ✗ {test_func['function_name']} - Not in Neo4j")
            else:
                failed_retrievals.append({
                    "function": test_func["function_name"],
                    "reason": "Not found in Milvus top results"
                })
                print(f"[{i+1}/{len(all_test_functions)}] ✗ {test_func['function_name']} - Not in Milvus top-{top_k}")

        # Print summary
        print("\n" + "="*60)
        print("RETRIEVAL TEST SUMMARY")
        print("="*60)
        print(f"Total Functions: {len(all_test_functions)}")
        print(f"Successfully Retrieved: {successful_retrievals}")
        print(f"Failed: {len(failed_retrievals)}")
        print(f"Success Rate: {successful_retrievals/len(all_test_functions)*100:.2f}%")

        if failed_retrievals:
            print(f"\nFailed Retrievals:")
            for failure in failed_retrievals:
                print(f"  • {failure['function']} - {failure['reason']}")

        print("="*60)

    finally:
        neo4j_driver.close()
        connections.disconnect("default")


def main():
    URI = "bolt://localhost:7687"
    AUTH = ("neo4j", "test")
    OUTPUT_FOLDER = "output"

    # Import data to Neo4j
    import_kg_files_to_neo4j(OUTPUT_FOLDER, URI, AUTH)

    # Import data to Milvus
    import_to_milvus(OUTPUT_FOLDER)

    # Query and visualize
    query_and_visualize(URI, AUTH)

    # Test retrieval
    test_retrieval(OUTPUT_FOLDER, URI, AUTH)
    query_and_visualize(URI, AUTH)


if __name__ == "__main__":
    main()