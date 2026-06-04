import torch
import pandas as pd


from module1.graph_builder import (
    CrossChainTemporalGraph
)

from module1.ego_subgraph import (
    EgoTemporalSubgraphExtractor
)



from module2.stage2_encoder import (
    Stage2Encoder
)



from module3.stage3_htgt import (
    Stage3Encoder
)



from module4.stage4_readout import (
    Stage4Readout
)



from module5.stage5_umgad import (
    Stage5UMGAD
)



from module6.stage6_ranking import (
    Stage6UnifiedScoring
)



df = pd.read_csv(
    "data/cross_chain_transactions.csv"
)

print(
    f"Loaded {len(df)} transactions"
)



graph_builder = (
    CrossChainTemporalGraph()
)

G = graph_builder.build_from_dataframe(
    df
)

print(
    "Nodes:",
    G.number_of_nodes()
)

print(
    "Edges:",
    G.number_of_edges()
)



extractor = (
    EgoTemporalSubgraphExtractor(G)
)

subgraphs = []

for node in G.nodes():

    S_v = extractor.extract_subgraph(

        target_node=node,

        k_hop=2,

        cutoff_time=float("inf")
    )

    subgraphs.append(S_v)

print(
    "Subgraphs:",
    len(subgraphs)
)

=

stage2 = Stage2Encoder(
    embedding_dim=128,
    heads=4
)



num_nodes = G.number_of_nodes()

node_type_ids = torch.randint(
    0,
    4,
    (num_nodes,)
)

chain_ids = torch.randint(
    0,
    6,
    (num_nodes,)
)

degree_features = torch.rand(
    num_nodes,
    1
)

relation_ids = torch.randint(
    0,
    4,
    (num_nodes,)
)

timestamps = torch.randint(
    0,
    1000,
    (num_nodes,)
)

stage2_embeddings = stage2(

    node_type_ids,

    chain_ids,

    degree_features,

    relation_ids,

    timestamps
)

print(
    "Stage2:",
    stage2_embeddings.shape
)


stage3 = Stage3Encoder(
    hidden_dim=128
)



node_embeddings = stage2_embeddings

print(
    "Stage3 input:",
    node_embeddings.shape
)



batch = torch.zeros(
    node_embeddings.shape[0],
    dtype=torch.long
)

stage4 = Stage4Readout(
    hidden_dim=128
)

stage4_output = stage4(
    node_embeddings,
    batch
)

aggregated_embedding = (

    stage4_output[
        "aggregated_embedding"
    ]
)

print(
    "Stage4:",
    aggregated_embedding.shape
)



stage5 = Stage5UMGAD()

stage5.fit(
    aggregated_embedding
)

stage5_output = stage5(
    aggregated_embedding
)

mahalanobis_scores = (

    stage5_output[
        "normalized_scores"
    ]
)

print(
    "Stage5:",
    mahalanobis_scores.shape
)


N = aggregated_embedding.shape[0]

bridge_features = torch.rand(
    N,
    3
)

temporal_features = torch.rand(
    N,
    3
)

node_ids = list(range(N))

stage6 = Stage6UnifiedScoring(

    alpha=0.5,

    beta=0.3,

    gamma=0.2
)

results = stage6(

    node_ids,

    mahalanobis_scores,

    bridge_features,

    temporal_features
)


TOP_K = 20

print(
    "\nTop Suspicious Nodes\n"
)

for rank in range(TOP_K):

    print(

        rank + 1,

        results[
            "ranked_nodes"
        ][rank],

        float(
            results[
                "ranked_scores"
            ][rank]
        )
    )