from collections import OrderedDict

import torch
from torch import nn

from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP
from model.modules.tcn import MultiScaleTCN


class AGFormerBlock(nn.Module):
    """
    Implementation of AGFormer block.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, 
                 mode='spatial', negative_attention=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode, negative_attention=negative_attention)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Dropout(p=drop_path)
        self.use_layer_scale = use_layer_scale

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """

        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MotionAGFormerBlock(nn.Module):
    """
    Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, n_frames=243, algatt_spatial=False, algatt_temporal=False):
        super().__init__()
        # ST Attention branch
        self.att_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale,
                                         mode='spatial', 
                                          negative_attention=algatt_spatial)
        self.att_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale,
                                          mode='temporal', 
                                           negative_attention=algatt_temporal)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """

        x_attn = self.att_temporal(self.att_spatial(x))
        
        return x_attn


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, qkv_bias=False, qkv_scale=None,  n_frames=243):
    """
    generates MotionAGFormer layers
    """
    layers = []
    for _ in range(n_layers):
        layers.append(MotionAGFormerBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          n_frames=n_frames))
    layers = nn.Sequential(*layers)

    return layers


class Transformers(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243):
        """
        :param n_layers: Number of layers.
        :param dim_in: Input dimension.
        :param dim_feat: Feature dimension.
        :param dim_rep: Motion representation dimension
        :param dim_out: output dimension. For 3D pose lifting it is set to 3
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.
        :param num_heads: Number of attention heads in attention branch
        :param qkv_bias: Whether to include bias in the linear layers that create query, key, and value or not.
        :param qkv_scale: scale factor to multiply after outer product of query and key. If None, it's set to
                          1 / sqrt(dim_feature // num_heads)
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param use_tcn: If true, uses MS-TCN for temporal part of the graph branch.
        :param graph_only: Uses GCN instead of GraphFormer in the graph branch.
        :param neighbour_num: Number of neighbors for temporal GCN similarity.
        :param n_frames: Number of frames. Default is 243
        """
        super().__init__()

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,

                                    n_frames=n_frames)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        
        x = self.joints_embed(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        
        if return_rep:
            return x
 

        x = self.head(x)

        return x

def _test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to(device)

    model = Transformers(dim_in=3, dim_out=3, num_joints=17).to(device)
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)
        # print(_)

    import time
    num_iterations = 100 
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")
    
    

    out = model(random_x)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
