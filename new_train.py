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
import argparse
parser = argparse.ArgumentParser(description='script usage')

parser.add_argument("--title", type=str, required=True)
parser.add_argument("--mod", type=str, required=True)
args = parser.parse_args()
project_title = args.title

#pr_version = '1'
#project_name = project_title+'_'+pr_version


# %%
cur = "c:/Users/COINSE/Downloads/simfl-extension"
os.chdir(cur)
os.chdir('d4j_data')
base = os.getcwd()
list_project = os.listdir()
list_project = [x for x in list_project if project_title in x]
os.chdir(cur)
list_project = sorted(list_project, key=lambda x: int(x.split('_')[1]), reverse=True)
m_scale = 0.25
#mutant_source = list_project[:int(m_scale*len(list_project))]
mutant_source = list_project[:3]
deprecated_bugs = ['Math_6', 'Math_38', 'Lang_2', 'Time_21']


# %%
# read data
dist = []
fm = {}
ms = {}
x = 0
mn = float('inf')
mx = 0
kt_n = 1
print('creating dataset')
for project_name in tqdm(list_project):
    if project_name in deprecated_bugs:
        continue
    if project_title in project_name and project_name in mutant_source:
        project = project_name.split('_')[0]
        project_version = project_name.split('_')[1]
        os.chdir(f'd4j_data_fix/{project_name}')
        with open('mutant_data_new.pkl', 'rb') as mf:
            mutant = pickle.load(mf)
        with open('test_data_new.pkl', 'rb') as tf:
            test = pickle.load(tf)
        with open('method_data_new.pkl', 'rb') as mef:
            method = pickle.load(mef)
        os.chdir(cur)
        r_dict = {}
        replacement_index = 0
        method_list = []
        for m in method:
            method_list.append(torch.from_numpy(method[m]['embedding']))
            r_dict[method[m]['method_name'].replace(" ", "")] = replacement_index
            replacement_index+=1
        label_tensor = torch.zeros(len(method_list))
        fm[project_name] = (torch.stack(method_list), label_tensor)
        for mutant_no in mutant:
            if mutant[mutant_no]['killer']:
                if args.mod == 'minimum':
                    ct = None
                    ctd = float('inf')
                    for t in mutant[mutant_no]['killer']:
                        d = np.linalg.norm(mutant[mutant_no]['embedding'] - test[t])
                        if d < ctd:
                            ctd = d
                            ct = t
                    tr = [torch.from_numpy(test[ct])]
                    if mutant[mutant_no]['signature'] in r_dict.keys():
                        ms[(project_name, mutant_no)] = (r_dict[mutant[mutant_no]['signature']], torch.from_numpy(mutant[mutant_no]['embedding']), tr)
                        x+=len(r_dict)
                if args.mod == 'all':
                    tl = []
                    tr = []
                    if len(mutant[mutant_no]['killer'])<=kt_n:
                        for t in mutant[mutant_no]['killer']:
                            tr.append(torch.from_numpy(test[t]))
                            tl.append(t)
                        if mutant[mutant_no]['signature'] in r_dict.keys():
                            ms[(project_name, mutant_no)] = (r_dict[mutant[mutant_no]['signature']], torch.from_numpy(mutant[mutant_no]['embedding']), tr)
                            x+=len(r_dict)*len(tr)
                        
                if args.mod == 'average':
                    tr = []
                    for t in mutant[mutant_no]['killer']:
                        tr.append(torch.from_numpy(test[t]))
                    tr = [torch.mean(torch.stack(tr), dim=0)]
                    if mutant[mutant_no]['signature'] in r_dict.keys():
                        ms[(project_name, mutant_no)] = (r_dict[mutant[mutant_no]['signature']], torch.from_numpy(mutant[mutant_no]['embedding']), tr)
                        x+=len(r_dict)
                if mutant[mutant_no]['signature'] in r_dict.keys() and len(tr)!=0:
                    if len(r_dict)*len(tr) < mn:
                        mn = len(r_dict)*len(tr)
                    if len(r_dict)*len(tr) > mx:
                        mx = len(r_dict)*len(tr)
           
        for m in method:
            if args.mod == 'minimum':
                dist.append(np.linalg.norm(method[m]['embedding'] - test[ct]))
            if args.mod == 'all':
                for t in tl:
                    dist.append(np.linalg.norm(method[m]['embedding'] - test[t]))
            if args.mod == 'average':
                dist.append(np.linalg.norm(method[m]['embedding'] - tr[0].numpy()))
print(len(ms))
print(x)
print(x / len(ms))
print(mn, mx)

# %%
from version_batch_modelloss import ContrastiveModel, ContrastiveLoss
from torch.utils.data import DataLoader, Dataset, Sampler
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

# %%
class PrecomputedBatchDataset(Dataset):
    def __init__(self, fix_method, mutant_sample):
        self.method = fix_method
        self.mutant = mutant_sample
        self.keys = list(mutant_sample.keys())

    def __len__(self):
        return len(self.mutant)

    def __getitem__(self, idx):
        k = self.keys[idx]
        mut = self.mutant[k]
        mutant_idx = mut[0]
        mutant_emb = mut[1]
        killing_test = mut[2]

        m_placeholder = self.method[k[0]][0].clone()
        b_size = m_placeholder.size()[0]

        method_tensor = m_placeholder.repeat(len(killing_test), 1) 
        method_tensor[torch.arange(len(killing_test)) * b_size + mutant_idx] = mutant_emb

        l_placeholder = self.method[k[0]][1].clone()
        label_tensor = l_placeholder.repeat(len(killing_test))
        label_tensor[torch.arange(len(killing_test)) * b_size + mutant_idx] = 1.0
        
        test_tensor = torch.stack(killing_test)
        test_tensor = test_tensor.repeat_interleave(b_size, dim=0) 

        #method_copy = [m_placeholder] * len(killing_test)
        #method_tensor = torch.stack(method_copy, dim=0)
        #for i in range(len(killing_test)):
        #    method_tensor[i*b_size+mutant_idx] = mutant_emb
        #
        #l_placeholder = self.method[k[0]][1].clone()
        #label_copy = [l_placeholder] * len(killing_test)
        #label_tensor = torch.stack(label_copy, dim=0)
        #for i in range(len(killing_test)):
        #    label_tensor[i*b_size+mutant_idx] = 1.0
        #
        #test_copy = [killing_test[0]] * b_size
        #test_tensor = torch.stack(test_copy, dim=0)
        #for i in range(1, len(killing_test)):
        #    test_copy = [killing_test[i]] * b_size
        #    test_tensor = torch.cat(test_tensor, torch.stack(test_copy, dim=0), dim=0)
        return method_tensor, test_tensor, label_tensor
    
def collate_fn(batch):
    return batch[0][0], batch[0][1], batch[0][2]

#torch.multiprocessing.set_start_method('fork', force = True)
#torch.multiprocessing.set_sharing_strategy('file_system')

dataset = PrecomputedBatchDataset(fm, ms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=collate_fn)
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True,
#    collate_fn=collate_fn, num_workers= 4, prefetch_factor = 2, persistent_workers = True)
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=collate_fn, persistent_workers = True, num_workers = 2)


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
arc = 'f'+str(len(mutant_source))
a = 0.1
m = args.mod if args.mod!='all' else args.mod+str(kt_n)
model = ContrastiveModel(embedding_dim=768, projection_dim=projection_dim, output_dim=output_dim, mode=m)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
loss = ContrastiveLoss(margin=init_margin)
optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate)
steps_per_epoch = len(ms)
total_steps = steps_per_epoch * expected_epoch
warmup_steps = int(0.1 * total_steps)
power = 2
def warmup_lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
model.train()
p_counter = 0
best_val_loss = float('inf')
loss_list = []
positive_loss_list = []
negative_loss_list = []

for epoch in range(num_epoch):
    epoch_loss = torch.tensor(0.0, device = device)
    positive_epoch_loss = torch.tensor(0.0, device = device)
    negative_epoch_loss = torch.tensor(0.0, device = device)
    train_time = []
    epoch_start = time.perf_counter()
    for batch_idx, (method_batch, test_batch, label) in tqdm(enumerate(dataloader)):
        method_batch = method_batch.to(device, non_blocking=True)
        test_batch = test_batch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        training_start = time.perf_counter()
        output = model(test_batch, method_batch)
        optimizer.zero_grad()
        l, pl, nl = loss(output, label)
        if torch.isnan(l).any():
            os.makedirs(f'new-model/{project_title}/version_batch', exist_ok=True)
            torch.save(model.state_dict(), f'new-model/{project_title}/version_batch/model_{arc}_lastnan.pth')
            exit()
        l.backward()
        optimizer.step()
        step = steps_per_epoch*epoch+batch_idx
        if step < warmup_steps:
            warmup_scheduler.step()
        epoch_loss += l
        positive_epoch_loss += pl
        negative_epoch_loss += nl
        training_end = time.perf_counter()
        train_time.append(training_end - training_start)
    epoch_end = time.perf_counter()
    grad_norm = compute_gradient_norm(model)
    avg_epoch_loss = (epoch_loss / len(ms)).item()
    positive_avg_epoch_loss = (positive_epoch_loss / len(ms)).item()
    negative_avg_epoch_loss = (negative_epoch_loss / len(ms)).item()
    
    loss_list.append(avg_epoch_loss)
    positive_loss_list.append(positive_avg_epoch_loss)
    negative_loss_list.append(negative_avg_epoch_loss)
    print(f'epoch {epoch+1} took: training - {sum(train_time)} seconds, total - {epoch_end - epoch_start} seconds')
    print(f'epoch {epoch+1} trained with {x} data, average loss:{avg_epoch_loss}, Gradient_norm:{grad_norm}, counter:{p_counter}')
    if step >= warmup_steps:
        plateau_scheduler.step(avg_epoch_loss)
    if avg_epoch_loss<best_val_loss:
        best_val_loss = avg_epoch_loss
        p_counter = 0
        os.makedirs(f'new-model/{project_title}/version_batch', exist_ok=True)
        torch.save(model.state_dict(), f'new-model/{project_title}/version_batch/model_{arc}_{a}_{m}_best.pth')
    else:
        p_counter+=1
    if p_counter >= 5:
        if epoch+1>30:
            break
    #if (epoch+1)%5==0:
    #    os.makedirs(f'new-model/{project_title}/version_batch', exist_ok=True)
    #    torch.save(model.state_dict(), f'new-model/{project_title}/version_batch/model_{arc}_{a}_{m}_{epoch+1}.pth')
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
#os.makedirs(f'new-model/{project_title}/version_batch', exist_ok=True)
#torch.save(model.state_dict(), f'new-model/{project_title}/version_batch/model_{arc}_{a}_{m}.pth')