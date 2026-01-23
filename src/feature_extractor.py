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
    """ Feature extractor using multi-headed attention mechanism. """
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)

        # Dimensions
        self.bb_dim = S.bb_total_dim
        self.yolo_dim = S.features_dim
        self.geo_dim = S.geo_dim

        # Token embedding dimension
        self.d_model = 128
        self.num_heads = 4

        # CNN dimensions
        self.cnn_final = 4

        # YOLO token
        self.yolo_net = nn.Sequential(
            nn.Linear(self.yolo_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model)
        )

        # BBoxes token
        self.bb_net = nn.Sequential(
            nn.Linear(self.bb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_model)
        )

        # Spatial/graph token
        self.geo_net = nn.Sequential(
            nn.Linear(self.geo_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model)
        )

        # 3D CNN
        self.img_conv = nn.Sequential(
            # kernel_size = (temporal, height, width)
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((S.stack_sz, self.cnn_final, self.cnn_final)),
        )

        # Projects to d_model size
        self.img_proj = nn.Linear(64 * self.cnn_final * self.cnn_final, self.d_model)

        # Temporal positional embeddings
        self.pos_emb = nn.Embedding(
            num_embeddings=S.stack_sz,
            embedding_dim=self.d_model
        )

        # Modality embeddings: 4 tokens per frame
        self.mod_emb = nn.Embedding(
            num_embeddings=4,
            embedding_dim=self.d_model
        )

        # Final projection to 256 dim
        self.final_linear = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        # Encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # CLS token
        self.cls_token = nn.Parameter(th.zeros(1,1, self.d_model))

    def forward(self, obs):
        bs = obs.shape[0]

        # Undo frame stacking
        obs = obs.view(bs, S.stack_sz, S.frame_dim)

        # Slice observations per timestep
        yolo_feats = obs[:, :, :S.features_dim]
        bb_feats   = obs[:, :, S.features_dim : S.features_dim + self.bb_dim]
        geo_feats  = obs[:, :, S.features_dim + self.bb_dim:S.features_dim + self.bb_dim + S.geo_dim]
        img_feats  = obs[:, :, S.features_dim + self.bb_dim + S.geo_dim:
                         S.features_dim + self.bb_dim + S.geo_dim + S.img_dim]

        # Run through CNN
        img_feats = img_feats.view(bs, 1, S.stack_sz, S.out_size, S.out_size)
        img_features = self.img_conv(img_feats)  # [bs, 64, stack_sz, 4, 4]

        # Reshape to get per-timestep features
        img_features = img_features.permute(0, 2, 1, 3, 4) 
        img_features = img_features.reshape(bs * S.stack_sz, -1)
        img_token = self.img_proj(img_features)
        img_token = img_token.view(bs, S.stack_sz, self.d_model)

        # Run remaining NNs 
        yolo_token = self.yolo_net(yolo_feats)
        bb_token = self.bb_net(bb_feats)
        geo_token = self.geo_net(geo_feats)

        # Setup temporal positional embeddings
        positions = th.arange(S.stack_sz, device=obs.device)
        pos = self.pos_emb(positions).unsqueeze(0)

        # Setup modality embeddings
        mod_ids = th.tensor([0, 1, 2, 3], device=obs.device)
        mod_embs = self.mod_emb(mod_ids)
        
        # Add embeddings
        yolo_token = yolo_token + pos + mod_embs[0]
        bb_token = bb_token + pos + mod_embs[1]
        geo_token = geo_token + pos + mod_embs[2]
        img_token = img_token + pos + mod_embs[3]

        # Build the token sequence
        tokens = th.cat([yolo_token, bb_token, geo_token, img_token], dim=1)

        # Add cls ttoken
        cls = self.cls_token.expand(bs, -1, -1)
        tokens = th.cat([cls, tokens], dim=1)

        # Transformer encoding
        encoded = self.encoder(tokens)

        # Extract CLS token
        state = encoded[:, 0]

        # Final projection for PPO
        return self.final_linear(state)