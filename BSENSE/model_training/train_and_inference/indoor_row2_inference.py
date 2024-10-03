import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bsense_dataset import BSenseDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils import *

row = 2
# Submitted batch job 3358101
indoor_file_list_data_path = "./file_list/experiment_list/test_file_list.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(indoor_file_list_data_path, "r") as f:
    file_path_list = f.readlines()


def custom_collate_fn(batch):
    # Unzip the batch
    data, doppler_data, seat_label, res_label, data_name = zip(*batch)
    
    # Concatenate along the first dimension
    data = torch.cat(data, dim=0)
    doppler_data = torch.cat(doppler_data, dim=0)
    seat_label = torch.cat(seat_label, dim=0)
    res_label = torch.cat(res_label, dim=0)
    # data_name = data_name
    return data, doppler_data, seat_label, res_label, data_name

bsense_dataset = BSenseDataset(file_path_list, row=row)

# Assuming `bsense_dataset.file_path_list` contains all the file paths
file_path_list = bsense_dataset.file_path_list

# Regular expression to extract the base path excluding the round part
path_regex = re.compile(r'(.+)_round_\d+$')

# Group file indices by base experiment path
experiment_groups = defaultdict(list)
for idx, file_path in enumerate(file_path_list):
    match = path_regex.match(file_path)
    if match:
        base_path = match.group(1)
        experiment_groups[base_path].append(idx)

print(len(bsense_dataset))

# Convert groups to a list to easily leave one out
experiment_group_list = list(experiment_groups.values())

bpm_errors = []
true_labels = []
pred_labels = []

for exp_id in range(0, 5):
    print(f"{exp_id}/5th experiment group training")

    # Load the pre-saved filenames for this fold
    # train_filenames = load_filenames_from_txt(f'./file_list/indoor_row2_file_list/train_filenames_fold_{exp_id}.txt')
    # val_filenames = load_filenames_from_txt(f'./file_list/indoor_row2_file_list/val_filenames_fold_{exp_id}.txt')
    test_filenames = load_filenames_from_txt(f'./file_list/indoor_row2_file_list/test_filenames_fold_{exp_id}.txt')

    ## Get the indices for the corresponding filenames
    # train_indices = get_indices(train_filenames, bsense_dataset)
    # val_indices = get_indices(val_filenames, bsense_dataset)
    test_indices = get_indices(test_filenames, bsense_dataset)

    # Define samplers using these indices
    # train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    # val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Define data loaders
    batch_size = 32
    # train_loader = DataLoader(bsense_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
    # val_loader = DataLoader(bsense_dataset, batch_size=64, sampler=val_sampler, collate_fn=custom_collate_fn)
    test_loader = DataLoader(bsense_dataset, batch_size=1, sampler=test_sampler, collate_fn=custom_collate_fn)

    best_f1 = -float("inf")
    best_test_loss = float("inf")
    # Load the model
    model_path = os.path.join('./model_weights/indoor_weights', 'whole_exp_{}_front_indoor_best_model_row{}.pth'.format(exp_id, row))
    model = torch.load(model_path, map_location=device if device != "cpu" else None)
    if row == 2:
        mask = [False, False, True, True]
    else:
        mask = [True, True, False, False]

    start_time = time.time()
    loss_fn = nn.BCEWithLogitsLoss()

    def one_hot_to_class(tensor):
        new_tensor = []
        for item in tensor:
            new_tensor.append(torch.argmax(item).item())
        new_tensor = torch.tensor(new_tensor)
        return new_tensor
    
    # evaluation
    model.eval()  # set the model to evaluation mode
    total_loss = 0

    pred_count_list = []
    label_count_list = []
    seat_loss1 = {}
    seat_count1 = {}
    # Apply sigmoid activation to get probabilities on the same scale as labels
    correct_predictions = 0
    total_predictions = 0
    lines = []
    seat_loss1[0] = 0
    seat_loss1[1] = 0
    seat_count1[0] = 0
    seat_count1[1] = 0
    with torch.no_grad():
        for i, (batch_data, batch_phase_data, batch_labels, batch_res_labels, data_name) in enumerate(test_loader):
            batch_data, batch_phase_data, batch_labels, batch_res_labels = batch_data.to(device), batch_phase_data.to(device), batch_labels.to(device), batch_res_labels.to(device)
            batch_data = batch_data.unsqueeze(1)
            outputs1, respiration_out1= model(batch_data, batch_phase_data)
            batch_labels = (batch_labels > 0).float()
            loss1 = loss_fn(outputs1, batch_labels)  # compute the loss
            vital_signs_loss1 = F.mse_loss(respiration_out1, batch_res_labels)
            respiration_out1[outputs1 < 0] = 0
            target_classes = one_hot_to_class(batch_labels)
            for index, target_class in enumerate(target_classes):
                target_class = int(target_class.item())
                if target_class not in seat_loss1:
                    seat_loss1[target_class] = 0
                    seat_count1[target_class] = 0
                seat_loss1[target_class] += F.l1_loss(respiration_out1[index], batch_res_labels[index])
                seat_count1[target_class] += 1
            total_loss += loss1.item()
            total_loss += vital_signs_loss1.item()
            # Convert outputs and batch labels to numpy arrays
            outputs1 = outputs1.cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()
            outputs1 = (outputs1 > 0.5).astype(int)
            
            outputs1_sum = np.sum(outputs1, axis=1)
            batch_labels_sum = np.sum(batch_labels, axis=1)
            outputs1_sum[outputs1_sum > 1] = 1
            batch_labels_sum[batch_labels_sum > 1] = 1
            outputs1_sum = smooth_signal(outputs1_sum)
            true_labels.append(batch_labels_sum)
            pred_labels.append(outputs1_sum)
            
            print(batch_labels_sum, outputs1_sum)

            for i, seat in enumerate(seat_loss1):
                if seat_count1[seat] == 0:
                    continue
                loss = seat_loss1[seat] / seat_count1[seat]
                bpm_errors.append(loss.item())
            print(data_name)
            
            
print("Avg BPM error:", np.mean(bpm_errors))
true_labels = np.concatenate(true_labels, axis=0)
pred_labels = np.concatenate(pred_labels, axis=0)
train_and_inference = accuracy_score(true_labels, pred_labels)
print(f"Detection Rate for 5 Experiments: {train_and_inference}")