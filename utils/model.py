
import torch
import torch.nn as nn

class CustomCrossAttention(nn.Module):
    def __init__(self, d_model_q, d_model_kv, d_k):
        super(CustomCrossAttention, self).__init__()
        self.d_k = d_k
        self.query_proj = nn.Linear(d_model_q, d_k)
        self.key_proj = nn.Linear(d_model_kv, d_k)
        self.value_proj = nn.Linear(d_model_kv, d_k)
        self.out_proj = nn.Linear(d_k, d_model_q)
        self.sigmoid = nn.Sigmoid()


    def forward(self, Q, KV, mask=None):
        Q_proj = self.query_proj(Q)
        K_proj = self.key_proj(KV)
        V_proj = self.value_proj(KV)
        attention_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V_proj)
        output = self.out_proj(attention_output)
        
        return output, attention_scores

    def parameters(self, **kwargs):
        return list(self.query_proj.parameters()) + list(self.key_proj.parameters())

    def train(self, train = True):
        self.query_proj.train(train)
        self.key_proj.train(train)




