import torch
import torch.nn as nn

#emb db
class EmbeddingDatabase:
    
    def __init__(self):

        self.embeddings = []

    def add(self,
            embeddings):

        self.embeddings.append(
            embeddings.detach()
        )

    def get_all(self):

        return torch.cat(
            self.embeddings,
            dim=0
        )

    def clear(self):

        self.embeddings = []
        
        #Guassian 
        class GaussianEstimator(
    nn.Module
):

    def __init__(
            self,
            eps=1e-6):

        super().__init__()

        self.eps = eps

        self.register_buffer(
            "mean",
            None
        )

        self.register_buffer(
            "covariance",
            None
        )

        self.register_buffer(
            "covariance_inv",
            None
        )

    def fit(
            self,
            embeddings):

        mean = embeddings.mean(
            dim=0
        )

        centered = (
            embeddings - mean
        )

        covariance = (

            centered.T @ centered

        ) / (

            embeddings.shape[0] - 1

        )

        covariance += (

            torch.eye(
                covariance.shape[0],
                device=embeddings.device
            )
            * self.eps

        )

        covariance_inv = (
            torch.linalg.inv(
                covariance
            )
        )

        self.mean = mean

        self.covariance = covariance

        self.covariance_inv = covariance_inv
        
        #mahala
        class MahalanobisScorer(
    nn.Module
):

    def __init__(
            self,
            gaussian_model):

        super().__init__()

        self.gaussian_model = (
            gaussian_model
        )

    def forward(
            self,
            embeddings):

        delta = (

            embeddings
            -
            self.gaussian_model.mean

        )

        distances = torch.sum(

            (delta @
             self.gaussian_model.covariance_inv)

            * delta,

            dim=1

        )

        return distances
    
    #norm
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
        
        #aggrg subG
        
        class AggregatedEmbeddingOutput(
    nn.Module
):

    def forward(
            self,
            embeddings):

        return embeddings
    
    class Stage5UMGAD(
    nn.Module
):

    def __init__(
            self,
            eps=1e-6):

        super().__init__()

        self.gaussian = (
            GaussianEstimator(
                eps
            )
        )

        self.scorer = (
            MahalanobisScorer(
                self.gaussian
            )
        )

        self.normalizer = (
            ScoreNormalizer()
        )

        self.output_layer = (
            AggregatedEmbeddingOutput()
        )

    def fit(
            self,
            embeddings):

        self.gaussian.fit(
            embeddings
        )

    def forward(
            self,
            embeddings):

        anomaly_scores = (

            self.scorer(
                embeddings
            )
        )

        normalized_scores = (

            self.normalizer(
                anomaly_scores
            )
        )

        return {

            "embeddings":
            self.output_layer(
                embeddings
            ),

            "mahalanobis_scores":
            anomaly_scores,

            "normalized_scores":
            normalized_scores
        }
        
        