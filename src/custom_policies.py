import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import MlpPolicy
from settings import S

class StopMLPPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=StopFeatureExtractor,
            **kwargs
        )

class StopFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space)
        
        # Dimensions from setting class 
        self.bb_dim = S.bb_total_dim
        self.yolo_dim = S.features_dim
        self.geo_dim = S.geo_dim

        # YOLO feature extractor network
        self.yolo_net = nn.Sequential(
            nn.Linear(self.yolo_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Bbox coordinate / size and classes network 
        self.bb_net = nn.Sequential(
            nn.Linear(self.bb_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        # Spatial information network        
        self.geo_net = nn.Sequential(
            nn.Linear(self.geo_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )

        # Fusion network
        fused_dim = 128 + 64 + 32
        self.fusion_net = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

    def forward(self, obs):
        bs = obs.shape[0]
        obs = obs.view(bs, S.stack_sz, S.frame_dim)

        # Slice the observation
        yolo_feats = obs[:, :, :S.features_dim]
        bb_feats = obs[:, :, S.features_dim : S.features_dim + self.bb_dim]
        geo_feats = obs[:, :, S.features_dim + self.bb_dim:]

        # Encode each feature group
        yolo_out = self.yolo_net(yolo_feats).mean(dim=1)
        bb_out = self.bb_net(bb_feats).mean(dim=1)
        geo_out = self.geo_net(geo_feats).mean(dim=1)

        fused = th.cat((yolo_out, bb_out, geo_out), dim=1)
        return self.fusion_net(fused)
