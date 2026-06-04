import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

#sub_G node embedd
class SubgraphNodeEmbeddingExtractor(
    nn.Module
):

    def __init__(self):

        super().__init__()

    def forward(
            self,
            node_embeddings):

        return node_embeddings
    
    #mean pooling
    
    class MeanPoolingReadout(
    nn.Module
):

    def __init__(self):

        super().__init__()

    def forward(
            self,
            node_embeddings,
            batch):

        z_v = global_mean_pool(
            node_embeddings,
            batch
        )

        return z_v
    
    #aggregate subG embedd
    
    class AggregatedSubgraphEmbedding(
    nn.Module
):

    def __init__(
            self,
            hidden_dim):

        super().__init__()

        self.projector = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ELU(),

            nn.LayerNorm(
                hidden_dim
            )
        )

    def forward(
            self,
            subgraph_embedding):

        return self.projector(
            subgraph_embedding
        )
        
        #enco
        
        class Stage4Readout(
    nn.Module
):

    def __init__(
            self,
            hidden_dim=128):

        super().__init__()

        self.node_embedding_extractor = (
            SubgraphNodeEmbeddingExtractor()
        )

        self.mean_pool = (
            MeanPoolingReadout()
        )

        self.aggregate = (
            AggregatedSubgraphEmbedding(
                hidden_dim
            )
        )

    def forward(
            self,
            node_embeddings,
            batch):

        z_nodes = (
            self.node_embedding_extractor(
                node_embeddings
            )
        )

        z_v = self.mean_pool(
            z_nodes,
            batch
        )

        z_bar_v = self.aggregate(
            z_v
        )

        return {

            "node_embeddings":
            z_nodes,

            "subgraph_embedding":
            z_v,

            "aggregated_embedding":
            z_bar_v
        }
        
        