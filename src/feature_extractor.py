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

        # Temporal positional embeddings
        self.pos_emb = nn.Embedding(
            num_embeddings=S.stack_sz,
            embedding_dim=self.d_model
        )

        # Modality embeddings: 3 tokens per frame
        self.mod_emb = nn.Embedding(
            num_embeddings=3,
            embedding_dim=self.d_model
        )

        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(self.d_model)

        # Final projection to 256 dim
        self.final_linear = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, obs):
        bs = obs.shape[0]

        # Undo frame stacking
        obs = obs.view(bs, S.stack_sz, S.frame_dim)

        # Slice observations per timestep
        yolo_feats = obs[:, :, :S.features_dim]
        bb_feats   = obs[:, :, S.features_dim : S.features_dim + self.bb_dim]
        geo_feats  = obs[:, :, S.features_dim + self.bb_dim:]

        # Per-modality embeddings [bs, T, d_model]
        yolo_token = self.yolo_net(yolo_feats)
        bb_token = self.bb_net(bb_feats)
        geo_token = self.geo_net(geo_feats)

        # Add temporal positional embeddings
        positions = th.arange(S.stack_sz, device=obs.device)
        pos = self.pos_emb(positions).unsqueeze(0)

        yolo_token = yolo_token + pos
        bb_token = bb_token + pos
        geo_token = geo_token + pos

        # Add modality embeddings
        yolo_token = yolo_token + self.mod_emb(th.tensor(0, device=obs.device))
        bb_token = bb_token + self.mod_emb(th.tensor(1, device=obs.device))
        geo_token = geo_token + self.mod_emb(th.tensor(2, device=obs.device))

        # Build the token sequence [bs, 3T, d_model] order: [y0, b0, g0, y1, b1, g1,]
        tokens = th.cat([yolo_token, bb_token, geo_token], dim=1)

        # Multi-head attention fusion
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = self.attn_norm(attn_out)

        # Pool across tokens 
        pooled = attn_out.mean(dim=1)

        # Final projection for PPO
        return self.final_linear(pooled)