"""Baseline experiment using our BSENSE method."""

import re
from pathlib import Path

import torch
from bsense_dataset import BSenseDataset
from rich import print
from torch.utils.data import DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

base_path = Path(__file__).parent

row = 1
test_file_path = base_path / "test_file.txt"
save_weights_path = base_path / "model_weight.pth"


def custom_collate_fn(batch):
    # Unzip the batch
    data, doppler_data, seat_label, res_label, data_name = zip(*batch)

    # Concatenate along the first dimension
    data = torch.cat(data, dim=0)
    doppler_data = torch.cat(doppler_data, dim=0)
    seat_label = torch.cat(seat_label, dim=0)
    res_label = torch.cat(res_label, dim=0)
    # data_name = torch.cat(data_name, dim=0)
    return data, doppler_data, seat_label, res_label, data_name


def extract_distance_angle(data_name):
    """Extract distance and angle from data point folder name.

    Since the dataset loader does not return the metadata file/information, we manully extract the distance and angle from the folder name.
    A folder name is like: "[person]Aligned_[radar config]_[radar profile]_distance1_degree_60_round_2", and we want "distance1" and "60".
    """
    if "distance2" in data_name:
        distance = 2.0
    elif "distance0.5" in data_name:
        distance = 0.5
    elif "distance1.5" in data_name:
        distance = 1.5
    else:
        distance = 1.0

    angle = re.search(r"degree_(\d+)", data_name).group(1)
    return distance, int(angle)


def bsense_baseline(test_file_path: Path, saved_weights_path: Path, row: int):
    """Baseline experiment using our BSENSE method. Inference on a test dataset.

    Args:
        test_file_path: Path to the test dataset. Each line is a path to a beamform-processed radar recording data.
        saved_weights_path: Path to the saved weights
        row: row number of the radar

    Returns:
        Dictionary that maps the data name to the following:
        distance: distance of the recording, measured from the chest to the radar
        angle: angle of the recording in degrees, where 0 degree is when the subject is facing the radar
        bpm_gt: Ground truth bpm
        bpm_radar: extracted bpm using range doppler"""
    # Initialize the test dataset
    with open(test_file_path, "r") as f:
        test_file_path_list = f.readlines()

    test_dataset = BSenseDataset(test_file_path_list, row=row)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn
    )

    # Load the model
    model = torch.load(
        saved_weights_path,
        map_location=device if device != "cpu" else None,
    )
    model.eval()

    # Random sample from test_loader
    result = {}
    for i, (
        batch_data,
        batch_phase_data,
        batch_labels,
        batch_res_labels,
        data_name,
    ) in enumerate(test_loader):
        batch_data, batch_phase_data, batch_labels, batch_res_labels = (
            batch_data.to(device),
            batch_phase_data.to(device),
            batch_labels.to(device),
            batch_res_labels.to(device),
        )
        batch_data = batch_data.unsqueeze(1)
        _, respiration_out1 = model(batch_data, batch_phase_data)
        # take the average of the first column
        predicted_respiration = respiration_out1.mean(dim=0)[0]
        distance, angle = extract_distance_angle(data_name[0])
        # print(f"Predicted: {respiration_out1.mean(dim=0)[0]}")
        # print(f"True: {batch_res_labels[0][0]}")
        result[data_name] = {
            "distance": distance,
            "angle": angle,
            "bpm_gt": batch_res_labels[0][0].item(),
            "bpm_radar": predicted_respiration.item(),
        }
    return result


if __name__ == "__main__":
    result = bsense_baseline(test_file_path, save_weights_path, row)
    print(result)
