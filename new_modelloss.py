import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim, projection_dim, output_dim, mode='euclidean'):
        super(ContrastiveModel, self).__init__()
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(projection_dim, output_dim)
        self.mode = mode
    
    def forward(self, embedding1, embedding2=None):
        x1 = self.projection(embedding1)
        x1 = self.layer_norm(x1)
        x1 = self.relu(x1)
        x1 = self.linear(x1)
        if embedding2 is not None:
            x2 = self.projection(embedding2)
            x2 = self.layer_norm(x2)
            x2 = self.relu(x2)
            x2 = self.linear(x2)
            if self.mode == 'cosine':
                distance = 1 - F.cosine_similarity(x1, x2, dim=1)
            else:
                distance = torch.norm(x1 - x2, p=2, dim=1)
            return distance 
        return x1

#blocked code for using bigger model, and progressive margin

#class ContrastiveModel(nn.Module):
#    def __init__(self, embedding_dim, projection_dim, output_dim, mode='euclidean', res_weight = 0.5):
#        super(ContrastiveModel, self).__init__()
#        
#        self.projection = nn.Linear(embedding_dim, projection_dim)
#        self.layer_norm = nn.LayerNorm(projection_dim)
#        self.relu = nn.ReLU()
#        self.gate_proj = nn.Linear(projection_dim, projection_dim)
#        self.sigmoid = nn.Sigmoid()
#        self.linear = nn.Linear(projection_dim, output_dim)
#        self.residual = res_weight
#        self.mode = mode
#    
#    def forward(self, embedding1, embedding2=None):
#        r1 = embedding1
#        x1 = self.projection(embedding1)
#        x1 = self.layer_norm(x1)
#        x1 = self.relu(x1)
#        g1 = self.sigmoid(self.gate_proj(x1))
#        x1 = x1 * g1
#        x1 = self.linear(x1)
#        x1 = x1 + self.residual*r1
#        if embedding2 is not None:
#            r2 = embedding2
#            x2 = self.projection(embedding2)
#            x2 = self.layer_norm(x2)
#            x2 = self.relu(x2)
#            g2 = self.sigmoid(self.gate_proj(x2))
#            x2 = x2 * g2
#            x2 = self.linear(x2)
#            x2 = x2 + self.residual*r2
#            if self.mode == 'cosine':
#                distance = 1 - F.cosine_similarity(x1, x2, dim=1)
#            else:
#                distance = torch.norm(x1 - x2, p=2, dim=1)
#            return distance 
#        return x1
        

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

#class ContrastiveLoss(nn.Module):
#    def __init__(self, init_margin=15.0, final_margin = 20.0, expected_epoch = 400):
#        super(ContrastiveLoss, self).__init__()
#        self.m_init = init_margin
#        self.m_final = final_margin
#        self.total_epochs = expected_epoch
#    
#    def get_margin(self, current_epoch):
#        return self.m_init * (1 - current_epoch / self.total_epochs) + self.m_final * (current_epoch / self.total_epochs)
#    
#    def forward(self, distances, labels, epoch):
#        dynamic_margin = self.get_margin(epoch)
#        dynamic_margin = self.m_init
#        positive_loss = labels * torch.pow(distances, 2)
#        negative_loss = (1 - labels) * torch.pow(torch.clamp(dynamic_margin - distances, min=0.0), 2)
#        loss = positive_loss + negative_loss
#        return torch.mean(loss), torch.mean(positive_loss), torch.mean(negative_loss)
        
