from collections import OrderedDict

import torch
from torch import nn
import numpy as np
from data.const import H36M_TO_MPI

import matplotlib.pyplot as plt

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]

def convert_h36m_to_mpi_connection():
    global connections
    new_connections = []
    for connection in connections:
        new_connection = (H36M_TO_MPI[connection[0]], H36M_TO_MPI[connection[1]])
        new_connections.append(new_connection)
    connections = new_connections


from model.modules.graph import GCN
# from model.modules.mlp import MLP
# from model.modules.tcn import MultiScaleTCN

class RevLinear(nn.Module):
    def __init__(self, in_features, out_features, A=None, bias = False):
        super(RevLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if A is None:
            self.A = nn.Parameter(torch.rand(out_features, in_features).T)
            
        else:
            self.A = A
            self.in_features, self.out_features = A.shape
            
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
    def forward(self, x, inverse=False):

        if not inverse:
            x = x @ self.A

            if self.bias is not None:
                x = x + self.bias
            return x
        else:
            if self.bias is not None:
                x = x - self.bias   
            x = x @ torch.linalg.pinv(self.A)
            return x
    def create_diff_matrix(n):
        def create_diff_line(n, i, j):
            t = torch.zeros(n)
            t[i] = 1
            t[j] = -1
            return t
        diff_matrix =  torch.stack([create_diff_line(n, i, j) for i in range(n) for j in range(i+1, n)]).T
        
        diff_matrix = torch.cat([diff_matrix, torch.ones(n).reshape(-1,1)/n], dim=1)
        
        return diff_matrix
        
        
class Linear(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """
    def __init__(self, dim_in, dim_out, num_joints, n_frames, n_layers=1, dropout=0.1, hidden=64):
        super(Linear, self).__init__()
        
        self.dimesion_layer = nn.Linear(dim_in, dim_out)
        self.joints_layer = nn.Linear(num_joints, num_joints)
        # self.temporal_layer = nn.Linear(n_frames, n_frames) 
        
        self.expend = nn.Linear(dim_in, 64)
        self.reduce = nn.Linear(64, dim_out)
        
        # self.A = RevLinear.create_diff_matrix(num_joints)
        # self.rev_lin = RevLinear(num_joints, 64, bias=False, A=self.A)
        
        self.rev_lin = RevLinear(num_joints, 32, bias=False)
        
        self.j = nn.Linear(self.rev_lin.A.shape[1], self.rev_lin.A.shape[1])
        
        # self.expend_d_layers = nn.ModuleList([nn.Linear(dim_in *(i+1), dim_out*(i+2)) for i in range(n_layers)])
        # self.expend_j_layers = nn.ModuleList([nn.Linear(num_joints *(i+1), num_joints*(i+2)) for i in range(n_layers)])
        # self.expend_t_layers = nn.ModuleList([nn.Linear(n_frames *(i+1), n_frames*(i+2)) for i in range(n_layers)])
        
        # self.reduce_d_layers = nn.ModuleList([nn.Linear(dim_out *(i+2), dim_out*(i+1)) for i in range(n_layers, 0, -1)])
        # self.reduce_j_layers = nn.ModuleList([nn.Linear(num_joints *(i+2), num_joints*(i+1)) for i in range(n_layers, 0, -1)])
        # self.reduce_t_layers = nn.ModuleList([nn.Linear(n_frames *(i+2), n_frames*(i+1)) for i in range(n_layers, 0, -1)])
        

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=27, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        
        # res = x  
        
        x = self.rev_lin(x.transpose(2, 3)).transpose(2, 3)
        
        res = x
        
        x = self.dimesion_layer(x)
        
        # x = x + res
        
        # x = self.joints_layer(x.transpose(2, 3)).transpose(2, 3)
        # x = self.temporal_layer(x.transpose(1, 3)).transpose(1, 3)
        
        x = self.j(x.transpose(2, 3)).transpose(2, 3)
        
        x = x + res
        
        x = self.rev_lin(x.transpose(2, 3), inverse=True).transpose(2, 3)
        

        
        # x = self.expend(x)
        # x = self.joints_layer(x.transpose(2, 3)).transpose(2, 3)
        # x = self.reduce(x)
        
        # x = res + x

        return x


def _test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to(device)

    model = Linear(dim_in=3, dim_out=3, num_joints=17).to(device)
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
