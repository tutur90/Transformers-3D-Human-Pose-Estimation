from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath


from model.modules.graph import GCN
# from model.modules.mlp import MLP
# from model.modules.tcn import MultiScaleTCN

class Linear(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """
    def __init__(self, dim_in, dim_out, num_joints, n_frames, n_layers=1, dropout=0.1, hidden=64):
        super(Linear, self).__init__()
        
        self.dimesion_layer = nn.Linear(dim_in, dim_out)
        self.joints_layer = nn.Linear(num_joints, num_joints)
        self.temporal_layer = nn.Linear(n_frames, n_frames)
        
        self.expend_d_layers = nn.ModuleList([nn.Linear(dim_in *(i+1), dim_out*(i+2)) for i in range(n_layers)])
        self.expend_j_layers = nn.ModuleList([nn.Linear(num_joints *(i+1), num_joints*(i+2)) for i in range(n_layers)])
        self.expend_t_layers = nn.ModuleList([nn.Linear(n_frames *(i+1), n_frames*(i+2)) for i in range(n_layers)])
        
        self.reduce_d_layers = nn.ModuleList([nn.Linear(dim_out *(i+2), dim_out*(i+1)) for i in range(n_layers, 0, -1)])
        self.reduce_j_layers = nn.ModuleList([nn.Linear(num_joints *(i+2), num_joints*(i+1)) for i in range(n_layers, 0, -1)])
        self.reduce_t_layers = nn.ModuleList([nn.Linear(n_frames *(i+2), n_frames*(i+1)) for i in range(n_layers, 0, -1)])
        

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=27, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        
        # x = self.u(x)
        # x = self.v(x.transpose(2, 3)).transpose(2, 3)
        
        x = self.dimesion_layer(x)
        x = self.joints_layer(x.transpose(2, 3)).transpose(2, 3)
        x = self.temporal_layer(x.transpose(1, 3)).transpose(1, 3)
        


        return x


def _test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to(device)

    model = MGCN(dim_in=3, dim_out=3, num_joints=17).to(device)
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
