import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RELATIONS = {
    "transfer":0,
    "swap":1,
    "bridge_in":2,
    "bridge_out":3
}

NODE_TYPES = {
    "EOA":0,
    "Exchange":1,
    "Bridge":2,
    "DeFi":3
}

CHAINS = {
    "Ethereum":0,
    "BSC":1,
    "Polygon":2,
    "Arbitrum":3,
    "Avalanche":4,
    "Ronin":5
}

#encoder

class NodeEncoder(nn.Module):
    
    def __init__(
            self,
            num_node_types,
            num_chains,
            embedding_dim):

        super().__init__()

        self.node_type_embedding = nn.Embedding(
            num_node_types,
            embedding_dim
        )

        self.chain_embedding = nn.Embedding(
            num_chains,
            embedding_dim
        )

        self.degree_projection = nn.Linear(
            1,
            embedding_dim
        )

    def forward(
            self,
            node_type_ids,
            chain_ids,
            degree_features):

        node_emb = self.node_type_embedding(
            node_type_ids
        )

        chain_emb = self.chain_embedding(
            chain_ids
        )

        degree_emb = self.degree_projection(
            degree_features
        )

        return (
            node_emb +
            chain_emb +
            degree_emb
        )
        
        #relation embedding tabl
        
        class RelationEncoder(nn.Module):
    
    def __init__(
            self,
            num_relations,
            embedding_dim):

        super().__init__()

        self.relation_embedding = nn.Embedding(
            num_relations,
            embedding_dim
        )

    def forward(
            self,
            relation_ids):

        return self.relation_embedding(
            relation_ids
        )
        
        #adaptive relation embedding
        
        class AdaptiveRelationEmbedding(nn.Module):
    
    def __init__(
            self,
            embedding_dim):

        super().__init__()

        self.W1 = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.W2 = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.gate = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(
            self,
            relation_embedding):

        transformed = torch.tanh(
            self.W1(relation_embedding)
        )

        gate = torch.sigmoid(
            self.gate(relation_embedding)
        )

        adaptive_embedding = (
            gate * transformed +
            (1-gate) *
            self.W2(relation_embedding)
        )

        return adaptive_embedding
    
    #temporal encodng
    
    class TemporalEncoding(nn.Module):
    
    def __init__(
            self,
            embedding_dim,
            max_len=50000):

        super().__init__()

        pe = torch.zeros(
            max_len,
            embedding_dim
        )

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2
            ) *
            (-math.log(10000.0)
             / embedding_dim)
        )

        pe[:,0::2] = torch.sin(
            position*div_term
        )

        pe[:,1::2] = torch.cos(
            position*div_term
        )

        self.register_buffer(
            "pe",
            pe
        )

    def forward(
            self,
            timestamps):

        return self.pe[
            timestamps
        ]
        
        #fusion layer
        
        class FeatureFusion(nn.Module):
    
    def __init__(
            self,
            embedding_dim):

        super().__init__()

        self.projector = nn.Linear(
            embedding_dim*3,
            embedding_dim
        )

    def forward(
            self,
            node_emb,
            relation_emb,
            temporal_emb):

        x = torch.cat(
            [
                node_emb,
                relation_emb,
                temporal_emb
            ],
            dim=-1
        )

        return self.projector(x)
    
    #initial transaformer attention
    
    class InitialAttentionBlock(nn.Module):
    
    def __init__(
            self,
            embedding_dim,
            num_heads):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim*4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_dim*4,
                embedding_dim
            )
        )

    def forward(self,x):

        attn_out,_ = self.attention(
            x,
            x,
            x
        )

        x = self.norm1(
            x + attn_out
        )

        ff_out = self.ffn(x)

        x = self.norm2(
            x + ff_out
        )

        return x
    
    class Stage2Encoder(nn.Module):
    
    def __init__(
            self,
            embedding_dim=128,
            heads=4):

        super().__init__()

        self.node_encoder = NodeEncoder(
            len(NODE_TYPES),
            len(CHAINS),
            embedding_dim
        )

        self.relation_encoder = RelationEncoder(
            len(RELATIONS),
            embedding_dim
        )

        self.adaptive_relation = (
            AdaptiveRelationEmbedding(
                embedding_dim
            )
        )

        self.temporal_encoder = (
            TemporalEncoding(
                embedding_dim
            )
        )

        self.fusion = FeatureFusion(
            embedding_dim
        )

        self.attention_block = (
            InitialAttentionBlock(
                embedding_dim,
                heads
            )
        )

    def forward(
            self,
            node_type_ids,
            chain_ids,
            degree_features,
            relation_ids,
            timestamps):

        node_emb = self.node_encoder(
            node_type_ids,
            chain_ids,
            degree_features
        )

        relation_emb = (
            self.relation_encoder(
                relation_ids
            )
        )

        relation_emb = (
            self.adaptive_relation(
                relation_emb
            )
        )

        temporal_emb = (
            self.temporal_encoder(
                timestamps
            )
        )

        x = self.fusion(
            node_emb,
            relation_emb,
            temporal_emb
        )

        x = self.attention_block(
            x
        )

        return x