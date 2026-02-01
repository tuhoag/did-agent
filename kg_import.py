import json
import os
from pathlib import Path
from neo4j import GraphDatabase


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


def main():
    URI = "bolt://localhost:7687"
    AUTH = ("neo4j", "test")
    OUTPUT_FOLDER = "output"

    # Import data
    # import_kg_files_to_neo4j(OUTPUT_FOLDER, URI, AUTH)

    # Query and visualize
    query_and_visualize(URI, AUTH)


if __name__ == "__main__":
    main()