# %%
import os
import json
from tqdm import tqdm
import time
import torch
import numpy as np
import pickle
import random

# %%
#project to be evaluated
pr_type = ['Chart', 'Math', 'Time', 'Lang']
project_title = pr_type[0]
#pr_version = '1'
#project_name = project_title+'_'+pr_version


# %%
start = time.time()
cur = "c:/Users/COINSE/Downloads/simfl-extension"
os.chdir(cur)
os.chdir('d4j_data')
base = os.getcwd()
list_project = os.listdir()
os.chdir(cur)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
# read data
# blocked code for data categorization
X_pos = {}
X_neg = {}
dist = []
method_list = set()
test_list = set()
method_test_set = set()
bf = 0
bp = 0
nb = 0
for project_name in list_project:
    if project_name == 'Math_38' or project_name == 'Math_6':
        continue
    if project_title in project_name:
        os.chdir(f'd4j_data_fix/{project_name}')
        with open('mutant_data_new.pkl', 'rb') as mf:
            mutant = pickle.load(mf)
        with open('test_data.pkl', 'rb') as tf:
            test = pickle.load(tf)
        os.chdir(cur)
        for mutant_no in mutant:
            for t in test:
                label = 0
                if t in mutant[mutant_no]['killer']:
                    label = 1
                    X_pos[(project_name, mutant_no, t)] = (mutant[mutant_no]['embedding'], test[t], label)
                else:
                    X_neg[(project_name, mutant_no, t)] = (mutant[mutant_no]['embedding'], test[t], label)
                dist.append(np.linalg.norm(mutant[mutant_no]['embedding']-test[t]))
                test_list.add(t)
                method_list.add(mutant[mutant_no]['method_name'])    
                method_test_set.add((mutant[mutant_no]['method_name'], t))
                #if label == 1:
                #    bf +=1
                #elif mutant[mutant_no]['tag'] == 'nb':
                #    nb +=1
                #else:
                #    bp +=1
dist = sorted(dist)
positive_sample_len = len(X_pos)
print(len(test_list))
print(len(method_list))
print(len(method_test_set))
print(len(X_pos))
print(len(X_neg))
#print(bf, bp, nb)

# %%
from new_modelloss import ContrastiveModel, ContrastiveLoss
from torch.utils.data import DataLoader, Dataset, Sampler
import matplotlib.pyplot as plt

# %%
#config
arc = 'relu'
mode = 'euclidean'
batch_size = 4096
num_epoch = 1000
expected_epoch = 100
projection_dim = 768
output_dim = 768
init_scale = 0.75
final_scale = 0.9
dist = sorted(dist)
init_margin = dist[int(init_scale*len(dist))]
#final_margin = dist[int(final_scale*len(dist))]
final_margin = dist[-1]
threshold = dist[0] / 2
learning_rate = 1e-3
res_weight = 1.0
print(threshold)
print(init_margin)
print(final_margin)

# %%
def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
        

# %%

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)
class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

        self.positive_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(dataset.labels) if label == 0]

    def __iter__(self):
        random.shuffle(self.positive_indices)
        random.shuffle(self.negative_indices)

        num_batches = min(len(self.positive_indices), len(self.negative_indices)) // self.half_batch

        for _ in range(num_batches):
            pos_batch = random.sample(self.positive_indices, self.half_batch)
            neg_batch = random.sample(self.negative_indices, self.half_batch)

            batch_indices = pos_batch + neg_batch
            random.shuffle(batch_indices)

            yield batch_indices
    def __len__(self):
        return min(len(self.positive_indices), len(self.negative_indices)) // self.half_batch


# %%
#blocked code for using bigger model, progressive margin, and learning rate management (not done during previous result)
ds_mod = ['zero']
m = 'euclidean'
for ds in ds_mod:
    X = []
    if ds == 'full':
        for key_pair in X_pos:
            X.append(X_pos[key_pair])
        for key_pair in X_neg:
            X.append(X_neg[key_pair])
    if m == 'cosine':
        margin = 1.0
        threshold = 0.5
    model = ContrastiveModel(embedding_dim=768, projection_dim=projection_dim, output_dim=output_dim, mode=m)
    loss = ContrastiveLoss(margin=init_margin)
    #model = ContrastiveModel(embedding_dim=768, projection_dim=projection_dim, output_dim=output_dim, mode=m, res_weight=res_weight)
    #loss = ContrastiveLoss(init_margin=init_margin, final_margin=final_margin, expected_epoch=expected_epoch)
    optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate)
    steps_per_epoch = len(X)/batch_size
    if ds != 'full':
        if ds == 'one':
            steps_per_epoch = 2 * positive_sample_len/batch_size
        else:
            steps_per_epoch = positive_sample_len/batch_size
    total_steps = steps_per_epoch * expected_epoch
    warmup_steps = int(0.1 * total_steps)
    power = 2
    def warmup_lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    #warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    #plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    model.to(device)
    model.train()
    p_counter = 0
    best_val_loss = float('inf')
    loss_list = []
    for epoch in range(num_epoch):
        method_test_set = set()
        if ds != 'full':
            X = []
            for key_pair in X_pos:
                X.append(X_pos[key_pair])
        if ds == 'one':
            neg_key_pair = list(X_neg.keys())
            random.shuffle(neg_key_pair)
            for key_pair in neg_key_pair:
                if len(X) < 2*positive_sample_len:
                    if (key_pair[1], key_pair[2]) not in method_test_set:
                        X.append(X_neg[key_pair])
                        method_test_set.add((key_pair[1], key_pair[2]))
            X_label = [x[2] for x in X]
            X_data = [x[:2] for x in X]
            dataset = CustomDataset(X_data, X_label)
            sampler = CustomSampler(dataset, batch_size)
            train_data = DataLoader(dataset, batch_sampler=sampler)
        else:
            X_label = [x[2] for x in X]
            X_data = [x[:2] for x in X]
            dataset = CustomDataset(X_data, X_label)
            train_data = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        epoch_loss = 0.0
        for batch_idx, (data, label) in enumerate(train_data):
            test = data[1].to(device)
            method = data[0].to(device)
            label = torch.Tensor(label)
            label = label.to(device)
            output = model(test, method)
            optimizer.zero_grad()
            l, _, _ = loss(output, label)
            #l, _, _ = loss(output, label, epoch)
            l.backward()
            optimizer.step()
            step = steps_per_epoch*epoch+batch_idx
            #if step < warmup_steps:
            #    warmup_scheduler.step()
            epoch_loss += l.item()
        grad_norm = compute_gradient_norm(model)
        avg_epoch_loss = epoch_loss / len(train_data)
        loss_list.append(avg_epoch_loss)
        print(f'epoch {epoch+1} trained with {len(X)} data, average loss:{avg_epoch_loss}, Gradient_norm:{grad_norm}')
        #if step >= warmup_steps:
        #    plateau_scheduler.step(avg_epoch_loss)
        if avg_epoch_loss<best_val_loss:
            best_val_loss = avg_epoch_loss
            p_counter = 0
        else:
            p_counter+=1
        if p_counter >= 10:
            if epoch+1>100:
                break
    epochs = list(range(1, len(loss_list)+1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_list, marker='o', linestyle='-', color='b', label='Training Loss')
    # Adding titles and labels
    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    os.makedirs(f'results/{project_title}', exist_ok=True)
    plt.savefig(f'results/{project_title}/{arc}_{ds}_newloss.png', format="png", dpi=300, bbox_inches="tight")
    plt.close()
    os.makedirs(f'new-model/{project_title}', exist_ok=True)
    torch.save(model.state_dict(), f'new-model/{project_title}/model_{arc}_{ds}.pth')

# %%



