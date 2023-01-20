import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_prob):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)
        # self.proj = nn.Sequential(
        #     nn.Linear(hidden_dims * 2, hidden_dims),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims, out_dims)
        # )

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output
