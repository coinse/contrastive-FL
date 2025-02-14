# %%
import os
import json
from tqdm import tqdm
import time
import torch
import numpy as np
import pickle
import random
import sys

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
#utility function from simfl-source
def get_failing_tests(project, fault_no, ftc_path):
    file_path = os.path.join(ftc_path, project, str(fault_no))
    ftc = []
    with open(file_path, "r") as ft_file:
        for test_case in ft_file:
            ftc.append(test_case.strip())
    return ftc

# %%
# read data
dist = []
batches = []
x = 0
mn = float('inf')
mx = 0
flag = True
for project_name in list_project:
    if project_name == 'Math_38' or project_name == 'Math_6':
        continue
    if project_title in project_name:
        project = project_name.split('_')[0]
        project_version = project_name.split('_')[1]
        FT_PATH = "./failing_tests"
        FAILING_TESTS = get_failing_tests(project, project_version, FT_PATH)
        os.chdir(f'd4j_data_fix/{project_name}')
        with open('mutant_data_new.pkl', 'rb') as mf:
            mutant = pickle.load(mf)
        with open('test_data_new.pkl', 'rb') as tf:
            test = pickle.load(tf)
        with open('method_data_new.pkl', 'rb') as mef:
            method = pickle.load(mef)
        os.chdir(cur)
        for mutant_no in mutant:
            if mutant[mutant_no]['killer']:
                dp = []
                ct = None
                ctd = float('inf')
                for t in mutant[mutant_no]['killer']:
                    d = np.linalg.norm(mutant[mutant_no]['embedding'] - test[t])
                    if d < ctd:
                        ctd = d
                        ct = t
                dp.append((mutant[mutant_no]['embedding'], test[ct], 1))
                for m in method:
                    if method[m]['method_name'] != mutant[mutant_no]['method_name']:
                        dp.append((method[m]['embedding'], test[ct], 0))
                        dist.append(np.linalg.norm(method[m]['embedding']- test[ct]))
                    if len(dp)>= 4096:
                        break
                random.shuffle(dp)
                batches.append(dp)
                x += len(dp)
                if len(dp)>mx:
                    mx = len(dp)
                if len(dp)<mn:
                    mn = len(dp)
                if flag:
                    print(sys.getsizeof(dp))
                    flag = False
print(len(batches))
print(x / len(batches))
print(mn, mx)

# %%
from version_batch_modelloss import ContrastiveModel, ContrastiveLoss
from torch.utils.data import DataLoader, Dataset, Sampler
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

# %%
class PrecomputedBatchDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches  # List of precomputed batches

    def __len__(self):
        return len(self.batches)  # Number of batches

    def __getitem__(self, idx):
        return self.batches[idx]  # Return batch directly
def collate_fn(batch):
    """Optimized collate function for DataLoader"""

    method_batch = torch.stack([torch.from_numpy(x[0]) for x in batch[0]], dim=0)
    test_batch = torch.stack([torch.from_numpy(x[1]) for x in batch[0]], dim=0)
    label = torch.tensor([x[2] for x in batch[0]], dtype=torch.float)
    
    return method_batch.pin_memory(), test_batch.pin_memory(), label.pin_memory()


dataset = PrecomputedBatchDataset(batches)
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=collate_fn)


# %%
#config
num_epoch = 1000
expected_epoch = 100
projection_dim = 768
output_dim = 768
init_scale = 0.75
final_scale = 0.9
dist = sorted(dist)
init_margin = dist[int(init_scale*len(dist))]
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
#blocked code for using bigger model, progressive margin, and learning rate management (not done during previous result)
arc = 'leaky_relu'
a = 0.1
m = 'euclidean'
model = ContrastiveModel(embedding_dim=768, projection_dim=projection_dim, output_dim=output_dim, mode=m)
loss = ContrastiveLoss(margin=init_margin)
optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate)
steps_per_epoch = len(batches)
total_steps = steps_per_epoch * expected_epoch
warmup_steps = int(0.1 * total_steps)
power = 2
def warmup_lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
model.to(device)
model.train()
p_counter = 0
best_val_loss = float('inf')
loss_list = []
positive_loss_list = []
negative_loss_list = []
print('training start')
for epoch in range(num_epoch):
    epoch_loss = 0.0
    positive_epoch_loss = 0.0
    negative_epoch_loss = 0.0
    for batch_idx, (method_batch, test_batch, label) in tqdm(enumerate(dataloader)):
        method_batch = method_batch.to(device, non_blocking=True)
        test_batch = test_batch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        output = model(test_batch, method_batch)
        optimizer.zero_grad()
        l, pl, nl = loss(output, label)
        l.backward()
        optimizer.step()
        step = steps_per_epoch*epoch+batch_idx
        if step < warmup_steps:
            warmup_scheduler.step()
        epoch_loss += l.item()
        positive_epoch_loss += pl.item()
        negative_epoch_loss += nl.item()
    grad_norm = compute_gradient_norm(model)
    
    avg_epoch_loss = epoch_loss / len(batches)
    positive_avg_epoch_loss = positive_epoch_loss / len(batches)
    negative_avg_epoch_loss = negative_epoch_loss / len(batches)
    
    loss_list.append(avg_epoch_loss)
    positive_loss_list.append(positive_avg_epoch_loss)
    negative_loss_list.append(negative_avg_epoch_loss)

    print(f'epoch {epoch+1} trained with {x} data, average loss:{avg_epoch_loss}, Gradient_norm:{grad_norm}')
    if step >= warmup_steps:
        plateau_scheduler.step(avg_epoch_loss)
    if avg_epoch_loss<best_val_loss:
        best_val_loss = avg_epoch_loss
        p_counter = 0
    else:
        p_counter+=1
    if p_counter >= 5:
        if epoch+1>30:
            break
    if epoch+1 % 5 == 0:
        torch.save(model.state_dict(), f'new-model/{project_title}/version_batch/model_{arc}_{a}_{m}_{epoch+1}.pth')
epochs = list(range(1, len(loss_list)+1))
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss_list, marker='o', linestyle='-', color='g', label='Training Loss')
plt.plot(epochs, positive_loss_list, marker='o', linestyle='-', color='b', label='Positive Training Loss')
plt.plot(epochs, negative_loss_list, marker='o', linestyle='-', color='r', label='Negative Training Loss')
# Adding titles and labels
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
os.makedirs(f'CROFL results/version_batch/{project_title}', exist_ok=True)
plt.savefig(f'CROFL results/version_batch/{project_title}/{arc}_newloss_{m}.png', format="png", dpi=300, bbox_inches="tight")
plt.close()
os.makedirs(f'new-model/{project_title}', exist_ok=True)
torch.save(model.state_dict(), f'new-model/{project_title}/version_batch/model_{arc}_{a}_{m}.pth')


