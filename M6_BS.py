import torch
import torch.nn as nn

#umgad scr 
class UMGADScoreAdapter(
    nn.Module
):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            mahalanobis_scores):

        return mahalanobis_scores
    
    class BridgeRiskScorer(
    nn.Module
):

    def __init__(
            self,
            input_dim=3):

        super().__init__()

        self.scorer = nn.Sequential(

            nn.Linear(
                input_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
            self,
            bridge_features):

        return self.scorer(
            bridge_features
        ).squeeze(-1)
        
        class TemporalRiskScorer(
    nn.Module
):

    def __init__(
            self,
            input_dim=3):

        super().__init__()

        self.scorer = nn.Sequential(

            nn.Linear(
                input_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
            self,
            temporal_features):

        return self.scorer(
            temporal_features
        ).squeeze(-1)
        
        class ScoreNormalizer(
    nn.Module
):

    def forward(
            self,
            scores):

        return (

            scores
            -
            scores.min()

        ) / (

            scores.max()
            -
            scores.min()
            +
            1e-12

        )
        
        class UnifiedAnomalyFusion(
    nn.Module
):

    def __init__(
            self,
            alpha=0.5,
            beta=0.3,
            gamma=0.2):

        super().__init__()

        assert abs(
            alpha+beta+gamma-1
        ) < 1e-6

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
            self,
            umgad_score,
            bridge_score,
            temporal_score):

        return (

            self.alpha
            *
            umgad_score

            +

            self.beta
            *
            bridge_score

            +

            self.gamma
            *
            temporal_score

        )
        
        class RankingModule:
    
    def rank(
            self,
            node_ids,
            scores):

        order = torch.argsort(

            scores,

            descending=True

        )

        ranked_nodes = [

            node_ids[i]

            for i in order

        ]

        ranked_scores = (
            scores[order]
        )

        return ranked_nodes, ranked_scores
    
    class Stage6UnifiedScoring(
    nn.Module
):

    def __init__(

            self,

            alpha=0.5,

            beta=0.3,

            gamma=0.2

    ):

        super().__init__()

        self.umgad_adapter = (
            UMGADScoreAdapter()
        )

        self.bridge_scorer = (
            BridgeRiskScorer()
        )

        self.temporal_scorer = (
            TemporalRiskScorer()
        )

        self.normalizer = (
            ScoreNormalizer()
        )

        self.fusion = (
            UnifiedAnomalyFusion(

                alpha,
                beta,
                gamma
            )
        )

        self.ranker = (
            RankingModule()
        )

    def forward(

            self,

            node_ids,

            mahalanobis_scores,

            bridge_features,

            temporal_features

    ):

        umgad = self.normalizer(

            self.umgad_adapter(
                mahalanobis_scores
            )
        )

        bridge = self.normalizer(

            self.bridge_scorer(
                bridge_features
            )
        )

        temporal = self.normalizer(

            self.temporal_scorer(
                temporal_features
            )
        )

        final_scores = (

            self.fusion(

                umgad,

                bridge,

                temporal
            )
        )

        ranked_nodes, ranked_scores = (

            self.ranker.rank(

                node_ids,

                final_scores
            )
        )

        return {

            "final_scores":
            final_scores,

            "ranked_nodes":
            ranked_nodes,

            "ranked_scores":
            ranked_scores,

            "umgad_score":
            umgad,

            "bridge_score":
            bridge,

            "temporal_score":
            temporal
        }
        
        