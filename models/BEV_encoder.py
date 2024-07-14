from math import sqrt

import torch
import torch.nn as nn

from models.mlp import MLP
from models.transformer import Perceiver
from utils.vector_utils import BEVObservation, VectorObservationConfig


class BEVEncoderConfig:
    model_dim: int = 256
    num_latents: int = 86
    num_blocks: int = 7
    num_heads: int = 8


class BEVEncoder(nn.Module):
    def __init__(
        self,
        encoder_config: BEVEncoderConfig,
        num_queries: int,
    ):
        super().__init__()

        model_dim = encoder_config.model_dim

        self.BEV_feature_encoder = MLP(
            BEVObservation.BEV_DIM, [model_dim], model_dim
        )
        
        self.perceiver = Perceiver(
            model_dim=model_dim,
            context_dim=model_dim,
            num_latents=encoder_config.num_latents,
            num_blocks=encoder_config.num_blocks,
            num_heads=encoder_config.num_heads,
            num_queries=num_queries,
        )

        self.out_features = model_dim


    def forward(self, ego_plan_query, context):
        batch = context.shape[0]
        device = context.device

        bev_token = self.BEV_feature_encoder(context)  # B x (200x200) x 256 tensor # Todo: add positional encoding]

        feat, _ = self.perceiver(ego_plan_query, bev_token)
        feat = feat.view(
            batch,
            self.perceiver.num_queries,
            feat.shape[-1],
        )

        return feat
