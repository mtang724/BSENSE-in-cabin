import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from vayyar_preprocess import (
    calculate_phase_shifts,
    compensate_phase_and_sum,
    preprocess_signal,
)


class BSenseDataset(Dataset):
    def __init__(self, file_path_list, row, transform=None):
        """
        Args:
            directory (string): Directory with all the files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_path_list = []
        self.row = row
        self.transform = transform
        for file_path in file_path_list:
            file_path = file_path.strip()
            doppler_path = os.path.join(file_path, "row{}_rd.npy".format(self.row))
            if not os.path.exists(doppler_path):
                print("Doppler file not found:", doppler_path)
                continue
            try:
                np.load(
                    os.path.join(file_path, "row{}_beamformed.npy".format(self.row))
                )
            except ValueError:
                print("Beamformed data corrupted:", file_path)
                continue
            doppler_data = torch.from_numpy(np.load(doppler_path)).float()
            # if doppler_data.shape[0] == 6:
            self.file_path_list.append(file_path)

    def __len__(self):
        return len(self.file_path_list)

    def pop(self, idx):
        self.file_path_list.pop(idx)

    def append(self, file_path):
        self.file_path_list.append(file_path)

    def split(self, ratio):
        # Random split
        np.random.shuffle(self.file_path_list)
        train_size = int(len(self.file_path_list) * ratio)
        # Split into two BSenseDataset objects
        train_dataset = BSenseDataset(
            self.file_path_list[:train_size], self.row, self.transform
        )
        test_dataset = BSenseDataset(
            self.file_path_list[train_size:], self.row, self.transform
        )
        return train_dataset, test_dataset

    def process_gt(self, json_data):
        seat_label = [0, 0, 0, 0]
        gt_label = [0, 0, 0, 0]
        # seat_mapping = ["driver", "passenger", "back_left", "back_right"]
        seat_mapping = ["middle"]
        for seat_name in json_data["occupied_seats"]:
            try:
                seat_idx = seat_mapping.index(seat_name)
            except Exception as e:
                print(e)
                continue
            if json_data["seats"][seat_name]["gt"] != None:
                gt_label[seat_idx] = float(json_data["seats"][seat_name]["gt"])
            if "baby_doll" in json_data["seats"][seat_name]["name"]:
                seat_label[seat_idx] = 1
            elif "child_doll" in json_data["seats"][seat_name]["name"]:
                seat_label[seat_idx] = 2
            else:
                seat_label[seat_idx] = 3
        return np.array(seat_label), np.array(gt_label)

    def return_agg_phase_signal(self, data, config):
        # config = np.load(os.path.join("/projects/bchl/mt55/bsense_data/data/0308Aligned/exp7_front_facing_infant_rear_left_child_right_20/50k_profile_1_distance2.4_gt_30_round_1", "config.npy"), allow_pickle=True).item()
        range_nfft = 128
        freqs = config["freq"]
        Ts = 1 / range_nfft / (freqs[1] - freqs[0] + 1e-16)  # Avoid nan checks
        time_vec = np.linspace(0, Ts * (range_nfft - 1), num=range_nfft)
        time_vec_inrange = time_vec[: data.shape[1]]
        time_vec_inrange = time_vec[: data.shape[1]]
        freq = 62e9
        tof_phase_shifts = calculate_phase_shifts(freq, time_vec_inrange)
        fs = 1 / config["sample_time"]
        phase_data = []
        # signals = data
        for instance in range(data.shape[0]):
            signals = data[instance]
            # print("signals shape:", signals.shape)
            # print("signals shape:", signals.shape)
            sumed_signal = compensate_phase_and_sum(signals, tof_phase_shifts)
            sumed_signal = np.angle(sumed_signal)
            sumed_signal = preprocess_signal(sumed_signal, fs)
            phase_data.append(sumed_signal)
        phase_data = np.array(phase_data)
        # print(sumed_signal.shape)
        return np.array(phase_data)

    def __getitem__(self, idx):
        rear_flag = False  # child is rear facing or not
        file_path = Path(self.file_path_list[idx].strip())
        json_path = file_path / "metadata.json"
        # Read the JSON file
        with open(json_path, "r") as file:
            json_data = json.load(file)

        data_name = json_data["exp_comment"]

        # Load the data
        data = np.load(file_path / "row{}_beamformed.npy".format(self.row))
        # print(data.shape)
        config = np.load(file_path / "config.npy", allow_pickle=True).item()
        data = self.return_agg_phase_signal(data, config)
        data = data.reshape(data.shape[0], 16, 16, data.shape[-1])
        data = torch.from_numpy(data).float()
        doppler_data = torch.from_numpy(
            np.load(file_path / "row{}_rd.npy".format(self.row))
        ).float()

        seat_label, res_label = self.process_gt(json_data)
        seat_label = torch.from_numpy(seat_label).float()
        seat_label = seat_label.unsqueeze(0).repeat(data.shape[0], 1)
        res_label = torch.from_numpy(res_label).float()
        res_label = res_label.unsqueeze(0).repeat(data.shape[0], 1)

        if not rear_flag:
            if self.row == 2:
                mask = [False, False, True, True]
            elif self.row == 1:
                mask = [True, True, False, False]
            elif self.row == 3:
                mask = [False, False, True, True]
                res_label[:, mask] = torch.FloatTensor(
                    [[0, 0] for _ in res_label[:, mask]]
                )
                seat_label[:, mask] = torch.FloatTensor(
                    [[0, 0] for _ in res_label[:, mask]]
                )

        if rear_flag:
            if self.row == 2:
                mask = [False, False, True, True]
                res_label[:, mask] = torch.FloatTensor(
                    [[0, 0] for _ in res_label[:, mask]]
                )
                seat_label[:, mask] = torch.FloatTensor(
                    [[0, 0] for _ in res_label[:, mask]]
                )
            elif self.row == 1:
                mask = [True, True, False, False]
            elif self.row == 3:
                mask = [False, False, True, True]

        return (data, doppler_data, seat_label[:, mask], res_label[:, mask], data_name)
