import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as Func
import math

class Model_PGCAN(nn.Module):
    
    def __init__(self, network, encoder = None  , data_params = None) -> None:
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.data_params = data_params

    def forward(self,X) :
        x = X[: , :1]
        y = X[: , 1:]

        out = self.encoder(x, y)

        out = self.network(input = X , features = out)

        return out

class Encoder(nn.Module):
    def __init__(self, 
                 n_features=128, 
                 res=[9, 9],  # Must have two values for 2D
                 n_cells=2, 
                 domain_size=[0, 2, 0, 1], 
                 mode='cosine', 
                 kernel_size=(3, 3), 
                 dtype=torch.float32,) -> None:
        super().__init__()
        self.mode = mode
        self.n_features = n_features
        self.res = res
        self.n_cells = n_cells
        self.domain_size = domain_size

        # Ensure kernel_size is a tuple of 2 ints
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif len(kernel_size) != 2:
            raise ValueError("kernel_size must be int or tuple/list of length 2")
        self.kernel_size = tuple(kernel_size)

        # Compute SAME padding
        padding = tuple((k - 1) // 2 for k in self.kernel_size)

        # Learnable field parameters
        a = [torch.rand(
                size=(self.n_cells, self.n_features, self.res[0], self.res[1]),
                dtype=dtype).uniform_(-1e-5, 1e-5)]
        self.F_active = nn.ParameterList(a)
        # self.F_active = nn.ParameterList([nn.Parameter(a)])???????????????

        # Depthwise convolution (groups = in_channels) with SAME padding
        self.conv_layer = nn.Conv2d(
            in_channels=self.n_features,
            out_channels=self.n_features,
            groups=self.n_features,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=False
        )

    def forward(self, x, y):
        features = []

        # Normalize coordinates to [-1, 1]
        x_min, x_max, y_min, y_max = self.domain_size
        x = (x - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)
        x = x * 2 - 1
        y = y * 2 - 1

        # Shape -> [n_cells, 1, H, W]
        x = torch.cat([x, y], dim=-1).unsqueeze(0).unsqueeze(0)
        x = x.repeat([self.n_cells, 1, 1, 1])

        for alpha in self.F_active:
            alpha = self.conv_layer(alpha)
            self.beta = alpha
            alpha = torch.tanh(alpha)

            # External function for grid sampling
            F = grid_sample_2d(alpha, x, step=self.mode, offset=True)
            dim = alpha.shape[1]
            features.append(F.sum(0).view(dim, -1).t())

        F = torch.cat(tuple(features), dim=1)
        return F

def grid_sample_2d(input, grid, step = 'cosine', offset = True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    ''' (iy,ix) will be the indices of the input
            1. normalize coordinates 0 to 1 (from -1 to 1)
            2. scaling to input size
            3. adding offset to make non-zero derivative interpolation '''
    ix = grid[..., 0]
    iy = grid[..., 1]
    if offset: # predefine the offset can accelerate training
        offset = torch.linspace(0,1-(1/N),N,device=grid.device).reshape(N,1,1)
        iy = ((iy+1)/2)*(IH-2) + offset
        ix = ((ix+1)/2)*(IW-2) + offset

    else:
        iy = ((iy+1)/2)*(IH-1)
        ix = ((ix+1)/2)*(IW-1)
    
    # compute corner indices
    with torch.no_grad():
        ix_left = torch.floor(ix)
        ix_right = ix_left + 1
        iy_top = torch.floor(iy)
        iy_bottom = iy_top + 1

    # compute weights
    dx_right = step_f(ix_right-ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom-iy)
    dy_top = 1 - dy_bottom

    nw = dx_right*dy_bottom
    ne = dx_left*dy_bottom
    sw = dx_right*dy_top
    se = dx_left*dy_top

    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_left, 0, IW-1, out=ix_left)
        torch.clamp(ix_right, 0, IW-1, out=ix_right)
        torch.clamp(iy_top, 0, IH-1, out=iy_top)
        torch.clamp(iy_bottom, 0, IH-1, out=iy_bottom)

    # look up values
    input = input.view(N, C, IH*IW)
    nw_val = torch.gather(input, 2, (iy_top * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    ne_val = torch.gather(input, 2, (iy_top * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))
    sw_val = torch.gather(input, 2, (iy_bottom * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    se_val = torch.gather(input, 2, (iy_bottom * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))

    # 2d_cosine/bilinear interpolation
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class NetworkM4_fused_disp(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, layers=[40, 40, 40, 40],
                 activation='tanh', init_method='xavier_uniform_'):
        super(NetworkM4_fused_disp, self).__init__()

        # Activation setup
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'gelu' : nn.GELU(),
            
        }
        self.activation = activation_dict[activation]
        # self.use_sin = (activation == 'sin')

        # Network layers
        self.H1 = nn.Linear(input_dim, layers[0])
        self.layers = nn.ModuleList()
        for width in layers:
            self.layers.append(nn.Linear(width, width))
        self.last = nn.Linear(layers[0], output_dim)

        # Store initialization method name
        self.init_method = init_method

        # Apply custom initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights
                init_fn = getattr(init, self.init_method, None)
                if init_fn is None:
                    raise ValueError(f"Unknown initialization method: {self.init_method}")
                init_fn(m.weight)

                # Initialize bias to non-zero using uniform distribution
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, 2*bound, 3*bound)

    def forward(self, input, features):
        F = features.shape[1] // 2
        U, V = features[:, :F], features[:, F:]

        H = torch.tanh(self.H1(input))
        
        for linear in self.layers:
            Z = linear(H)
            Z = self.activation(Z) 
            Z = torch.sigmoid(Z)
            H = (1 - Z) * U + Z * V

        out = self.last(H)
        return out
    
class NetworkM4_fused_rho(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, layers=[40, 40, 40, 40],
                 activation='tanh', init_method='xavier_uniform_', tau = 1.0):
        super(NetworkM4_fused_rho, self).__init__()

        # Activation setup
        activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'gelu': nn.GELU()
        }
        self.activation = activation_dict[activation]
        # self.use_sin = (activation == 'sin')

        # Network layers
        self.H1 = nn.Linear(input_dim, layers[0])
        self.layers = nn.ModuleList()
        for width in layers:
            self.layers.append(nn.Linear(width, width))
        self.last = nn.Linear(layers[0], output_dim)

        # Store initialization method name
        self.init_method = init_method

        # Store initialization method name
        self.tau = tau

        # Apply custom initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights
                init_fn = getattr(init, self.init_method, None)
                if init_fn is None:
                    raise ValueError(f"Unknown initialization method: {self.init_method}")
                init_fn(m.weight)

                # Initialize bias to non-zero using uniform distribution
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, 2*bound, 3*bound)

    def forward(self, input, features):
        F = features.shape[1] // 2
        U, V = features[:, :F], features[:, F:]

        H = torch.tanh(self.H1(input))
        
        for linear in self.layers:
            Z = linear(H)
            Z = self.activation(Z) 
            Z = torch.sigmoid(Z)
            H = (1 - Z) * U + Z * V

        out = Func.softmax(self.last(H)/self.tau, dim=1)
        return out




