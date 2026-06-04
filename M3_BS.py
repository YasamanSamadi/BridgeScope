import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#relation aware

class RelationAwareQKV(nn.Module):
    
    def __init__(
            self,
            hidden_dim,
            num_relations):

        super().__init__()

        self.query = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.key = nn.ModuleList([

            nn.Linear(
                hidden_dim,
                hidden_dim
            )

            for _ in range(num_relations)
        ])

        self.value = nn.ModuleList([

            nn.Linear(
                hidden_dim,
                hidden_dim
            )

            for _ in range(num_relations)
        ])

    def forward(
            self,
            node_features,
            relation_ids):

        q = self.query(
            node_features
        )

        k_list = []
        v_list = []

        for i,r in enumerate(relation_ids):

            k_list.append(
                self.key[r](
                    node_features[i]
                )
            )

            v_list.append(
                self.value[r](
                    node_features[i]
                )
            )

        k = torch.stack(k_list)

        v = torch.stack(v_list)

        return q,k,v
    
    #temporal decay func
    class TemporalDecay(nn.Module):
    
    def __init__(
            self,
            decay_rate=0.1):

        super().__init__()

        self.decay_rate = decay_rate

    def forward(
            self,
            delta_t):

        return torch.exp(
            -self.decay_rate * delta_t
        )
        
        #edge hete atten
        
        class HeterogeneousEdgeAttention(nn.Module):
    
    def __init__(
            self,
            hidden_dim,
            num_relations):

        super().__init__()

        self.qkv = RelationAwareQKV(
            hidden_dim,
            num_relations
        )

        self.temporal_decay = (
            TemporalDecay()
        )

        self.scale = math.sqrt(
            hidden_dim
        )

    def forward(
            self,
            node_embeddings,
            relation_ids,
            delta_times):

        q,k,v = self.qkv(
            node_embeddings,
            relation_ids
        )

        scores = (
            q*k
        ).sum(-1)

        scores = (
            scores / self.scale
        )

        decay = self.temporal_decay(
            delta_times
        )

        scores = scores * decay

        alpha = F.softmax(
            scores,
            dim=0
        )

        messages = (
            alpha.unsqueeze(-1)
            * v
        )

        return messages, alpha
    
    #relation aggregateor
    
    class RelationLevelAggregator(nn.Module):
    
        def __init__(self,
            hidden_dim,
            num_relations):

            super().__init__()

        self.num_relations = (
            num_relations
        )

        self.hidden_dim = (
            hidden_dim
        )

    def forward(
            self,
            messages,
            relation_ids):

        device = messages.device

        relation_output = torch.zeros(

            self.num_relations,
            self.hidden_dim,

            device=device
        )

        for r in range(
                self.num_relations):

            mask = (
                relation_ids == r
            )

            if mask.sum() > 0:

                relation_output[r] = (

                    messages[mask]
                    .mean(0)

                )

        return relation_output
    
    
    #relation fusion layer
    class RelationFusion(nn.Module):
    
        def __init__(
            self,
            hidden_dim,
            num_relations):

            super().__init__()

        self.attn = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
            self,
            relation_embeddings):

        scores = self.attn(
            relation_embeddings
        )

        scores = F.softmax(
            scores.squeeze(-1),
            dim=0
        )

        z = (

            scores.unsqueeze(-1)
            *
            relation_embeddings

        ).sum(0)

        return z
    
    #one het tempo layer
    
    class HeterogeneousTemporalLayer(
    nn.Module
):

        def __init__(
            self,
            hidden_dim,
            num_relations,
            dropout=0.2):

            super().__init__()

        self.edge_attention = (

            HeterogeneousEdgeAttention(
                hidden_dim,
                num_relations
            )
        )

        self.relation_agg = (

            RelationLevelAggregator(
                hidden_dim,
                num_relations
            )
        )

        self.relation_fusion = (

            RelationFusion(
                hidden_dim,
                num_relations
            )
        )

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
            self,
            h,
            relation_ids,
            delta_times):

        messages, alpha = (

            self.edge_attention(
                h,
                relation_ids,
                delta_times
            )
        )

        relation_embeddings = (

            self.relation_agg(
                messages,
                relation_ids
            )
        )

        z = self.relation_fusion(
            relation_embeddings
        )

        out = self.linear(z)

        out = F.elu(out)

        out = self.dropout(out)

        out = self.norm(
            out + h.mean(0)
        )

        return out
    
    #multi-lay temp GNN encoder
    class HeterogeneousTemporalGNN(
    nn.Module
):

        def __init__(
            self,
            hidden_dim=128,
            num_relations=4,
            num_layers=3):

            super().__init__()

        self.layers = nn.ModuleList([

            HeterogeneousTemporalLayer(

                hidden_dim,
                num_relations

            )

            for _ in range(
                num_layers
            )
        ])

    def forward(
            self,
            node_embeddings,
            relation_ids,
            delta_times):

        h = node_embeddings

        layer_outputs = []

        for layer in self.layers:

            h_global = layer(

                h,
                relation_ids,
                delta_times
            )

            h = h + h_global

            layer_outputs.append(
                h
            )

        return h, layer_outputs
    
    
    class Stage3Encoder(
    nn.Module
):

        def __init__(
            self,
            hidden_dim=128,
            num_relations=4):

            super().__init__()

        self.temporal_gnn = (

            HeterogeneousTemporalGNN(

                hidden_dim=hidden_dim,

                num_relations=num_relations,

                num_layers=3
            )
        )

    def forward(
            self,
            stage2_embeddings,
            relation_ids,
            delta_times):

        node_embeddings, all_layers = (

            self.temporal_gnn(

                stage2_embeddings,

                relation_ids,

                delta_times
            )
        )

        return {

            "node_embeddings":
            node_embeddings,

            "layer_outputs":
            all_layers
        }