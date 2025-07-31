import os
from getpass import getpass

import args

from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
import dataloader
import argparse
from data_chunk import run_chunk
from utils import *

def creat_metagraph(args, content, gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()
    whole_chunk = content

    if args.grained_chunk == True:
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-simple', action='store_true')
    parser.add_argument('-construct_graph', action='store_true')
    parser.add_argument('-inference',  action='store_true')
    parser.add_argument('-grained_chunk',  action='store_true')
    parser.add_argument('-trinity', action='store_true')
    parser.add_argument('-trinity_gid1', type=str)
    parser.add_argument('-trinity_gid2', type=str)
    parser.add_argument('-ingraphmerge',  action='store_true')
    parser.add_argument('-crossgraphmerge', action='store_true')
    parser.add_argument('-dataset', type=str, default='koreabank')
    parser.add_argument('-data_path', type=str, default=("C:/Users/leejuhan/IdeaProjects/_Graphrag/dataset/koreabank.csv"))
    parser.add_argument('-test_data_path', type=str, default='./dataset_ex/report_0.txt')
    args = parser.parse_args()

    url=os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
    username=os.getenv("NEO4J_USERNAME", "neo4j")
    password=os.getenv("NEO4J_PASSWORD", "12345678")

    n4j = Neo4jGraph(
        url=url,
        username=username,             # Default username
        password=password     # Replace 'yourpassword' with your actual password
    )

    file_path = args.data_path
    content = dataloader.load_high(file_path)
    gid = str_uuid()
    creat_metagraph(args, content, gid, n4j)

if __name__ == "__main__":
    main()
