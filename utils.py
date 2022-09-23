import torch
from torch import nn

embed_len = 50
hidden_dim = 128
n_layers=2
vocab = torch.load("vocab.pt")
class RNNClassifier(nn.Module):
    def __init__(self):
        super(RNNClassifier, self).__init__()
        self.seq = nn.Sequential(nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len))
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.lstm = nn.LSTM(input_size = embed_len, hidden_size = hidden_dim, num_layers = n_layers, batch_first = True, bidirectional = True)
        #self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(2 * hidden_dim, 128),
                                    nn.ReLU(),
                                    # nn.Linear(128, 256),
                                    # nn.ReLU(),
                                    # nn.Linear(256, 128),
                                    # nn.ReLU(),
                                    nn.Linear(128, 6),
                                    nn.Sigmoid())
        

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.lstm(embeddings, (torch.randn(2 * n_layers, len(X_batch), hidden_dim, device = "cuda"), torch.randn(2 * n_layers, len(X_batch), hidden_dim, device = "cuda")))
        return self.linear(output[:,-1])