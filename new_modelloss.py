import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim, projection_dim, output_dim, mode='euclidean'):
        super(ContrastiveModel, self).__init__()
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(projection_dim, output_dim)
        self.mode = mode
    
    def forward(self, embedding1, embedding2=None):
        x1 = self.projection(embedding1)
        x1 = self.layer_norm(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.linear(x1)
        if embedding2 is not None:
            x2 = self.projection(embedding2)
            x2 = self.layer_norm(x2)
            x2 = self.relu(x2)
            x2 = self.dropout(x2)
            x2 = self.linear(x2)
            if self.mode == 'cosine':
                distance = 1 - F.cosine_similarity(x1, x2, dim=1)
            elif self.mode == 'dot':
                distance = torch.sum(x1 * x2, dim=1) 
            else:
                distance = torch.norm(x1 - x2, p=2, dim=1)
            return distance 
        return x1

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=15.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distances, labels):
        dynamic_margin = self.margin
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(dynamic_margin - distances, min=0.0), 2)
        loss = positive_loss + negative_loss
        return torch.mean(loss), torch.mean(positive_loss), torch.mean(negative_loss)
