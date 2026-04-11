import torch
import torch.nn as nn

class OwnGRUCell(nn.Module):
    """
    Custom GRU Cell implemented from scratch.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(OwnGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for:
        # Update gate (z), Reset gate (r), Candidate hidden state (h_tilde)
        # We concatenate for z and r: 2 * hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 2 * hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(2 * hidden_size))

        # Weights for Candidate hidden state
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hn = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        nn.init.orthogonal_(self.W_in)
        nn.init.orthogonal_(self.W_hn)
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)
        nn.init.zeros_(self.b_in)
        nn.init.zeros_(self.b_hn)

    def forward(self, x, h_prev):
        # x: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)

        # 1. Gate calculation (Update z and Reset r)
        gates = (torch.matmul(x, self.W_ih) + self.b_ih) + (torch.matmul(h_prev, self.W_hh) + self.b_hh)
        z_gate, r_gate = gates.chunk(2, 1)

        z_gate = torch.sigmoid(z_gate)
        r_gate = torch.sigmoid(r_gate)

        # 2. Candidate hidden state (n or h_tilde)
        # Note: reset gate is applied to h_prev before multiplication or inside the weight mult
        # Standard PyTorch implementation: n = tanh(W_in*x + b_in + r * (W_hn*h_prev + b_hn))
        n_gate = torch.tanh(
            (torch.matmul(x, self.W_in) + self.b_in) + 
            r_gate * (torch.matmul(h_prev, self.W_hn) + self.b_hn)
        )

        # 3. New hidden state
        h_next = (1 - z_gate) * n_gate + z_gate * h_prev

        return h_next
