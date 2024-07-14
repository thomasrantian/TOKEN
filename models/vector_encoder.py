from math import sqrt

import torch
import torch.nn as nn

from models.mlp import MLP
from models.transformer import Perceiver
from utils.vector_utils import VectorObservation, VectorObservationConfig


class VectorEncoderConfig:
    model_dim: int = 256
    num_latents: int = 2
    num_blocks: int = 1
    num_heads: int = 1


class VectorEncoder(nn.Module):
    def __init__(
        self,
        encoder_config: VectorEncoderConfig,
        observation_config: VectorObservationConfig,
        num_queries: int,
    ):
        super().__init__()

        model_dim = encoder_config.model_dim
        self.ego_vehicle_encoder = MLP(
            VectorObservation.EGOINFO_DIM, [model_dim], model_dim
        )
        self.agents_encoder = MLP(
            VectorObservation.AGENTSINFO_DIM, [model_dim], model_dim
        )
        self.lane_geom_encoder = MLP(
            VectorObservation.LANESGEOM_DIM, [model_dim], model_dim
        )
        self.lane_topo_encoder = MLP(VectorObservation.LANESTOPO_DIM, [model_dim], model_dim)
        # self.route_embedding = nn.Parameter(
        #     torch.randn((observation_config.num_route_points, model_dim))
        #     / sqrt(model_dim)
        # )

        self.perceiver = Perceiver(
            model_dim=model_dim,
            context_dim=model_dim,
            num_latents=encoder_config.num_latents,
            num_blocks=encoder_config.num_blocks,
            num_heads=encoder_config.num_heads,
            num_queries=num_queries,
        )

        self.out_features = model_dim

    def forward(self, obs: VectorObservation):
        batch = obs.lane_geom_descriptors.shape[0]
        device = obs.lane_geom_descriptors.device

        # route_token = self.route_embedding[None] + self.route_encoder(
        #     obs.route_descriptors
        # )
        agents_token = self.agents_encoder(obs.agents_descriptors)
        lane_geom_token = self.lane_geom_encoder(obs.lane_geom_descriptors)
        lane_topo_token = self.lane_topo_encoder(obs.lane_topo_descriptors)
        context = torch.cat((agents_token, lane_geom_token, lane_topo_token), -2)
        context_mask = torch.cat(
            (
                obs.agents_descriptors[:, :, 0] != 0,  # agents
                torch.ones(
                    (batch, lane_geom_token.shape[1]), device=device, dtype=bool
                ),  # lane_geom
                torch.ones(
                    (batch, lane_topo_token.shape[1]), device=device, dtype=bool
                ),  # lane_topo_token
            ),
            dim=1,
        )

        ego_vehicle_state = obs.ego_vehicle_descriptor.float()
        ego_vehicle_feat = self.ego_vehicle_encoder(ego_vehicle_state)

        feat, _ = self.perceiver(ego_vehicle_feat, context, context_mask=context_mask)
        feat = feat.view(
            batch,
            self.perceiver.num_queries,
            feat.shape[-1],
        )

        return feat
