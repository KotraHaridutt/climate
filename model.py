import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # The core difference: Convolutions instead of Linear layers
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, 
                              out_channels=4 * hidden_dim, 
                              kernel_size=kernel_size, 
                              padding=padding)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        # Concatenate input and previous hidden state along the channel axis
        combined = torch.cat([x, h_cur], dim=1)  
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ExtremeWeatherModel(nn.Module):
    def __init__(self, input_channels=5, hidden_dim=16, kernel_size=3):
        super(ExtremeWeatherModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # The Spatio-Temporal feature extractor
        self.convlstm_cell = ConvLSTMCell(input_channels, hidden_dim, kernel_size)
        
        # Final layer to collapse the hidden state into a single probability per pixel
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.size()
        
        # Initialize hidden states with zeros
        device = x.device
        h_state = torch.zeros(b, self.hidden_dim, h, w).to(device)
        c_state = torch.zeros(b, self.hidden_dim, h, w).to(device)
        
        # Unroll the sequence over time
        for time_step in range(t):
            h_state, c_state = self.convlstm_cell(x[:, time_step, :, :, :], (h_state, c_state))
            
        # We only care about predicting the future right after the sequence, 
        # so we use the final hidden state to make our prediction.
        out = self.final_conv(h_state)
        return out # Note: No sigmoid here! We use BCEWithLogitsLoss which applies it safely.