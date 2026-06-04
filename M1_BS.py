import pandas as pd
import numpy as np
import networkx as nx

from collections import defaultdict
from typing import Dict, List

#relation type
RELATION_TYPES = {
    "transfer": 0,
    "swap": 1,
    "bridge_in": 2,
    "bridge_out": 3
}

#node type
NODE_TYPES = {
    "EOA": 0,
    "Exchange": 1,
    "Bridge": 2,
    "DeFi": 3
}

#graph constructor
class CrossChainTemporalGraph:
    
    def __init__(self):

        self.graph = nx.MultiDiGraph()

        self.node_metadata = {}

        self.edge_metadata = []



    def add_node(
            self,
            address,
            node_type,
            chain):

        if address not in self.graph:

            self.graph.add_node(
                address,
                node_type=node_type,
                chain=chain
            )

            self.node_metadata[address] = {

                "node_type": node_type,
                "chain": chain
            }



    def add_edge(
            self,
            sender,
            receiver,
            tx_hash,
            timestamp,
            amount,
            chain,
            relation):

        self.graph.add_edge(

            sender,
            receiver,

            tx_hash=tx_hash,
            timestamp=timestamp,
            amount=amount,
            chain=chain,
            relation=relation
        )

        self.edge_metadata.append({

            "source": sender,
            "target": receiver,
            "timestamp": timestamp,
            "amount": amount,
            "chain": chain,
            "relation": relation,
            "tx_hash": tx_hash
        })


    def build_from_dataframe(self, df):

        df = df.sort_values("timestamp")

        for _, row in df.iterrows():

            self.add_node(
                row["from"],
                row["from_type"],
                row["chain"]
            )

            self.add_node(
                row["to"],
                row["to_type"],
                row["chain"]
            )

            self.add_edge(
                sender=row["from"],
                receiver=row["to"],
                tx_hash=row["hash"],
                timestamp=row["timestamp"],
                amount=row["amount"],
                chain=row["chain"],
                relation=row["relation"]
            )

        return self.graph
    
    #temporal edge index
    
    class TemporalIndexer:
    
    def __init__(self, graph):

        self.graph = graph

    def build_temporal_index(self):

        temporal_edges = []

        for u, v, key, data in self.graph.edges(
                keys=True,
                data=True):

            temporal_edges.append({

                "u": u,
                "v": v,
                "timestamp": data["timestamp"],
                "relation": data["relation"]
            })

        temporal_edges.sort(
            key=lambda x: x["timestamp"]
        )

        return temporal_edges
    
    
    #Ego-Temporal Subgraph Extraction
    
    class EgoTemporalSubgraphExtractor:
    
    def __init__(self, graph):

        self.graph = graph

    

    def get_k_hop_nodes(
            self,
            target,
            k):

        visited = {target}

        frontier = {target}

        for _ in range(k):

            next_frontier = set()

            for node in frontier:

                nbrs = set(
                    self.graph.successors(node)
                )

                nbrs.update(
                    self.graph.predecessors(node)
                )

                next_frontier.update(nbrs)

            next_frontier -= visited

            visited.update(next_frontier)

            frontier = next_frontier

        return visited

    

    def extract_subgraph(
            self,
            target_node,
            k_hop,
            cutoff_time):

        nodes = self.get_k_hop_nodes(target_node,k_hop)

        subgraph = nx.MultiDiGraph()

        for n in nodes:

            subgraph.add_node(
                n,
                **self.graph.nodes[n]
            )

        for u, v, key, data in self.graph.edges(
                keys=True,
                data=True):

            if u not in nodes:
                continue

            if v not in nodes:
                continue

            if data["timestamp"] > cutoff_time:
                continue

            subgraph.add_edge(u,v,**data)

        return subgraph
    
    
    #generate temporal snapshot
    
    class TemporalSnapshotGenerator:
    
    def __init__(self, graph):

        self.graph = graph

    def create_snapshots(self):

        timestamps = []

        for _, _, _, data in self.graph.edges(
                keys=True,
                data=True):

            timestamps.append(
                data["timestamp"]
            )

        timestamps = sorted(
            list(set(timestamps))
        )

        snapshots = {}

        for t in timestamps:

            Gt = nx.MultiDiGraph()

            for node in self.graph.nodes():

                Gt.add_node(
                    node,
                    **self.graph.nodes[node]
                )

            for u, v, key, data in self.graph.edges(
                    keys=True,
                    data=True):

                if data["timestamp"] <= t:

                    Gt.add_edge(
                        u,
                        v,
                        **data
                    )

            snapshots[t] = Gt

        return snapshots
    
    #training samples
    
    class TrainingSampleBuilder:
    
    def __init__(self,graph):

        self.graph = graph

    def build_samples(
            self,
            k_hop=2):

        samples = []

        extractor = EgoTemporalSubgraphExtractor(
            self.graph
        )

        for node in self.graph.nodes():

            times = []

            for _, _, _, data in self.graph.edges(
                    keys=True,
                    data=True):

                if data["timestamp"] not in times:

                    times.append(
                        data["timestamp"]
                    )

            for t in times:

                subgraph = extractor.extract_subgraph(
                    node,
                    k_hop,
                    t
                )

                samples.append({

                    "target_node": node,
                    "timestamp": t,
                    "subgraph": subgraph
                })

        return samples
    
    
    df = pd.read_csv("cross_chain_transactions.csv")

graph_builder = CrossChainTemporalGraph()

G = graph_builder.build_from_dataframe(df)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

indexer = TemporalIndexer(G)

temporal_edges = indexer.build_temporal_index()

snapshot_generator = TemporalSnapshotGenerator(G)

snapshots = snapshot_generator.create_snapshots()

extractor = EgoTemporalSubgraphExtractor(G)

subgraph = extractor.extract_subgraph(
    target_node="0xABC",
    k_hop=2,
    cutoff_time=1700000000
)

sample_builder = TrainingSampleBuilder(G)

samples = sample_builder.build_samples(k_hop=2)