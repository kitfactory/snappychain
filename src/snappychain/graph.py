import os
from py2neo import Graph


neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")


def get_graph():
    graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))
    return graph

if __name__ == "__main__":
    graph = get_graph()
    print(graph)

