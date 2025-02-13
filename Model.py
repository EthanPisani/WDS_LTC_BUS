import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.v = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, h):
        # h: [batch_size, sequence_length, hidden_size]
        W_h = torch.matmul(h, self.W)  # [batch_size, sequence_length, hidden_size]
        U_h = torch.matmul(h, self.U)  # [batch_size, sequence_length, hidden_size]
        attention_scores = torch.matmul((W_h + U_h), self.v)  # [batch_size, sequence_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, sequence_length, 1]
        return attention_weights

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=1, dropout=0.0, use_attention=True):
        super(ImprovedLSTM, self).__init__()

        # Encoder
        self.use_attention = use_attention
        self.cnn = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            for _ in range(1)
        ])
        if self.use_attention:
            self.attention_layer = AttentionLayer(hidden_size * 2)  # Bidirectional, so double the hidden_size

        # Decoder
        self.pristup_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder_layer = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Layer normalization
        self.layer_norm = nn.LayerNormalization(hidden_size)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        batch_size, sequence_length, _ = x.size()

        # CNN encoder
        x = x.permute(0, 2, 1)  # [batch_size, input_size, sequence_length]
        x = self.cnn(x)  # [batch_size, hidden_size, sequence_length]
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, hidden_size]
        x = self.dropout(x)

        # LSTM encoder
        for lstm in self.layers:
            x, (hn, cn) = lstm(x)

        # Attention
        if self.use_attention:
            attention_weights = self.attention_layer(x)
            x = x * attention_weights
            x = torch.sum(x, dim=1)  # [batch_size, hidden_size * 2]
        else:
            x = x[:, -1, :]  # [batch_size, hidden_size * 2]

        x = F.relu(self.pristup_layer(x))  # Add non-linearity
        x = x.unsqueeze(1).repeat(1, sequence_length, 1)  # [batch_size, sequence_length, hidden_size]

        # LSTM decoder
        x, (hn, cn) = self.decoder_layer(x)
        x = self.output_layer(x)

        return x  # [batch_size, sequence_length, output_size]

# Example usage:
if __name__ == '__main__':
    # Hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 1
    num_layers = 2
    dropout = 0.1
    sequence_length = 24
    batch_size = 32

    # Instantiate the model
    model = ImprovedLSTM(input_size=input_size, hidden_size=hidden_size,
                        output_size=output_size, num_layers=num_layers,
                        dropout=dropout, use_attention=True)

    # Test the model
    x = torch.randn(batch_size, sequence_length, input_size)  # [32, 24, 1]
    output = model(x)  # [32, 24, 1]
    print(output.shape)
