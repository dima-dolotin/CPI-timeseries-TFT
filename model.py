import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class TFTConfig:
    n_past =        # context
    n_fut =
    dim_past =
    dim_fut =
    model_dim =                 # should be divisable by n_head
    lstm_hid_dim = 4 * model_dim 
    n_head = 

class GRN(nn.Module):
    def __init__(self, config, in_d=None ,out_d=None):
        super().__init__()
        if in_d is None:
            in_d = config.model_dim
        if out_d is None:
            out_d = config.model_dim

        self.linear1 = nn.Linear(in_d, out_d)              
        self.linear2 = nn.Linear(out_d, out_d)
        self.linear_glu = nn.Linear(out_d, 2*out_d)

        self.elu = nn.ELU()
        self.norm = nn.LayerNorm(out_d)

        # For GRN implementation in VarSelection block, where in_d = Cxd, out_d = C
        if in_d != out_d:
            self.downsample = nn.Linear(in_d, out_d, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, a):
        nu = self.linear1(a)
        nu = self.elu(nu)
        nu = self.linear2(nu)
        nu = self.linear_glu(nu) 
        v, gate = nu.chunk(2, dim=-1)
        nu = gate.sigmoid() * v
        
        return self.norm(nu + self.downsample(a))


class GLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.model_dim
        self.linear_glu = nn.Linear(d, 2*d)

    def forward(self, x):
        x = self.linear_glu(x) # (B, T, C) -> (B, T, 2*C)
        v, gate = x.chunk(2, dim=-1)
        x = gate.sigmoid() * v # (B, T, C)

        return x
    

class AddNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.model_dim
        self.norm = nn.LayerNorm(d)

    def forward(self, y, x):
        return self.norm(y + x)


class VarSelection(nn.Module):
    def __init__(self, config, input_dim):   # input_dim should be changed for historical vs future inputs
        super().__init__()
        d = config.model_dim

        self.grn_flat = GRN(config, in_d = input_dim * d, out_d = input_dim)
        self.embd_list = nn.ModuleList([nn.Linear(1, d) for _ in range(input_dim)])
        self.grn_list = nn.ModuleList([GRN(config) for _ in range(input_dim)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.size() # batch size, time point, embedding dimensionality

        xi = []
        for i in range(C):
            x_ = x[:, :, i:i+1] # (B, T, 1)
            x_ = self.embd_list[i](x_) # (B, T, d)
            xi.append(x_) # (B, T, C, d)
        xi = torch.stack(xi, dim=2) # (B, T, C, d)

        cap_xi = xi.flatten(start_dim=2) # (B, T, Cxd)
        v = self.grn_flat(cap_xi) # (B, T, Cxd) ->  (B, T, C)
        v = self.softmax(v)

        proc_xi = []
        for j in range(C):
            xi_ = xi[:, :, j, :] # (B, T, d)
            xi_ = self.grn_list[j](xi_) # (B, T, d)
            proc_xi.append(xi_) # (B, T, C, d)
        proc_xi = torch.stack(proc_xi, dim=2) # (B, T, C, d)

        fin_xi = v.unsqueeze(-1) * proc_xi
        fin_xi = torch.sum(fin_xi, dim=2) # (B, T, d)

        return fin_xi


class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config.model_dim
        self.h = config.lstm_hid_dim

        # Input, forget, output and new memory LSTM cells concatenated
        self.input_w = nn.Linear(self.d, 4 * self.h)
        self.hidden_w = nn.Linear(self.h, 4 * self.h)

        # Dimensionality reduction h -> C
        self.proj = nn.Linear(self.h, self.d)

    def forward(self, x):
        B, T, C = x.size() # batch size, time point, embedding dimensionality

        # Initial states
        c = [torch.zeros(B, self.h, device=x.device, dtype=x.dtype)] # (B, h)
        h = [torch.zeros(B, self.h, device=x.device, dtype=x.dtype)] # (B, h)
        
        for i in range(T):
            x_ = x[:, i, :] # (B, C)
            h_ = h[i] # (B, h)

            # Unilateral pass forward through the linear layer for all cells
            x_input = self.input_w(x_) # (B, C) -> (B, 4*h)
            x_hidden = self.hidden_w(h_) # (B, h) -> (B, 4*h)
            x_input_gate = x_input + x_hidden # (B, 4*h)

            # Chunk the concatenated vector for each cell
            x_input, x_forget, x_output, x_memory = x_input_gate.chunk(4, dim=-1) 
            
            #Apply functionals for each cell
            x_input = x_input.sigmoid() # (B, h)
            x_forget = x_forget.sigmoid() # (B, h)
            x_output = x_output.sigmoid() # (B, h)
            x_memory = x_memory.tanh() # (B, h)

            c.append(x_forget * c[i] + x_input * x_memory) # (B, T+1, h)
            h.append(x_output * c[i+1].tanh()) # (B, T+1, h)

        # Make tensor, remove zero vector, reduce dimensions back to C
        h = torch.stack(h, dim=1)[:, 1:, :] # (B, T, h)
        h = self.proj(h) # (B, T, h) -> (B, T, C)
        return h
        
class InterpretableAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        d = config.model_dim
        wide_contxt = config.n_past + config.n_fut

        self.linear_attn = nn.Linear(d, 3 * d)
        self.linear_proj = nn.Linear(d, d)

        self.register_buffer('tril', torch.tril(torch.ones(wide_contxt, wide_contxt))
                             .view(1, 1, wide_contxt, wide_contxt))

    def forward(self, x):
        B, T, C = x.size() # batch size, time point, embedding dimensionality

        q, k, v  = self.linear_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        attn = (q @ k.transpose(2,3)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = attn.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        y = attn.mean(dim=1) @ v # (B, T, T) x (B, T, C) -> (B, T, C)
        return self.linear_proj(y)


class TFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_past = config.dim_past
        self.dim_fut = config.dim_fut
        d = config.model_dim

        # Variable selection
        self.varselect_past = VarSelection(config, self.dim_past)
        self.varselect_fut = VarSelection(config, self.dim_fut)

        self.norm = AddNorm(config)
        
        # Temporal layers
        self.lstm = LSTM(config)
        self.attn = InterpretableAttention(config)

        # Gating
        self.gate_lstm = GLU(config)
        self.gate_attn = GLU(config)
        self.gate_out = GLU(config)

        # Feed-forward layers
        self.grn_pre_attn = GRN(config)
        self.grn_post_attn = GRN(config)

        # Quantile output layer
        self.lin_quant = nn.Linear(d, 3)

    def forward(self, x_past, x_fut):
        B, Tp, Cp = x_past.size()
        B, Tf, Cf = x_fut.size()

        # Variable selection and LSTM
        x_past = self.varselect_past(x_past) # (B, Tp, C)
        x_fut = self.varselect_fut(x_fut) # (B, Tf, C)
        x = torch.cat((x_past, x_fut), dim=1) # (B, T, C)
        y = self.lstm(x) # (B, T, C)
        y = self.gate_lstm(y) # (B, T, C)

        # Temporal Fusion decoder
        x = self.norm(y, x) # (B, T, C)
        x_fut = x[:, Tp:, :] # (B, Tf, C) residual connection input

        x = self.grn_pre_attn(x) # (B, T, C)
        y = self.attn(x)[:, Tp:, :] # (B, Tf, C)
        y = self.gate_attn(y) # (B, Tf, C)

        # Position-wise feed-forward and output layers
        x = self.norm(y, x[:, Tp:, :]) # (B, Tf, C)
        y = self.grn_post_attn(x) # (B, Tf, C)
        y = self.gate_out(y) # (B, Tf, C)
        y = self.norm(y, x_fut) # (B, Tf, C)

        # Quantile outputs
        pred = self.lin_quant(y) # (B, Tf, 3)

        return pred # Return the quantile values in one tensor


# Loss function ==============================================

def quant_loss(y, pred, q):
    return torch.max(q*(y - pred), (1.0 - q)*(pred - y))

def tft_loss(y, pred):
    loss_q1 = quant_loss(y, pred[:, :, 0], 0.1).mean()
    loss_q2 = quant_loss(y, pred[:, :, 1], 0.5).mean()
    loss_q3 = quant_loss(y, pred[:, :, 2], 0.9).mean()
    return (loss_q1 + loss_q2 + loss_q3)/3
        
