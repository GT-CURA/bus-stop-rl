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
        super().__init__(observation_space, features_dim=128)

        # Per-frame feature dims
        self.bb_dim = S.bbs_kept * (S.bb_dim + S.num_classes)

        # YOLO Feature extractor network
        self.yolo_net = nn.Sequential(
            nn.Linear(S.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Bbox coordinate / size and classes network
        self.bb_net = nn.Sequential(
            nn.Linear(self.bb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Spatial information network
        self.geo_net = nn.Sequential(
            nn.Linear(S.geo_dim, 16),
            nn.ReLU()
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 32 + 16, 128),
            nn.ReLU()
        )

    def forward(self, obs):
        frame_dim = S.features_dim + self.bb_dim + S.geo_dim
        obs = obs.view(-1, S.stack_sz, frame_dim)

        # Pull each section of observation out 
        yolo_feats = obs[:, :, :S.features_dim]
        bb_feats = obs[:, :, S.features_dim : S.features_dim + self.bb_dim]
        geo_feats = obs[:, :, S.features_dim + self.bb_dim:]

        # Run each through their NN
        yolo_out = self.yolo_net(yolo_feats).mean(dim=1)
        coord_out = self.bb_net(bb_feats).mean(dim=1)
        geo_out = self.geo_net(geo_feats).mean(dim=1)

        # Run output through fused NN
        fused = th.cat((yolo_out, coord_out, geo_out), dim=1)
        return self.fusion_net(fused)