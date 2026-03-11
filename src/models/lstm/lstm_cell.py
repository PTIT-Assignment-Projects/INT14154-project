import torch
import torch.nn as nn
class OwnLSTMCell(nn.Module):
    """
    Custom LSTM Cell implemented from scratch.
    """

    def __init__(self, input_size, hidden_size):
        super.__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for gates: input (i), forget (f), cell (g/c_tilde), output (o)
        # We concatenate them for efficiency: 4 * hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        nn.init.orthogonal_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, x, states):
        # x: (batch_size, input_size)
        # states: (h_prev, c_prev) where each is (batch_size, hidden_size)
        h_prev, c_prev = states

        # Combined gate calculation
        gates = (torch.matmul(x, self.W_ih) + self.b_ih) + (torch.matmul(h_prev, self.W_hh) + self.b_hh)

        # Split into individual gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next