"""Baseline signal processing method using range doppler."""

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from rich import print
from scipy.signal import find_peaks

from metadata import MetadataManager
from util import ProcessRadarSlidingWindow


def plot_peaks(gt, peaks):
    plt.clf()
    plt.plot(gt)
    plt.plot(peaks, gt[peaks], "x")
    plt.savefig("bsense_baseline_gt_peaks.png")


def doppler_baseline(data_dir: str, plot=False):
    """Baseline signal processing method using range doppler. Takes the path to one data recording.

    Returns:
        distance: distance of the recording, measured from the chest to the radar
        angle: angle of the recording in degrees, where 0 degree is when the subject is facing the radar
        bpm_gt: Ground truth bpm
        bpm_radar: extracted bpm using range doppler
    """
    assert Path(data_dir).is_dir(), f"Invalid data directory: {data_dir}"
    assert (Path(data_dir) / "recording.npy").exists()
    assert (Path(data_dir) / "config.npy").exists()
    assert (Path(data_dir) / "metadata.json").exists()

    # Extract distance and angle from metadata
    metadata = MetadataManager("metadata.json")
    metadata.read(data_dir)
    distance = metadata["distance"]
    angle = metadata["degree"]

    if not metadata["gt_valid"] and not metadata["aligned"] and not metadata["has_gt"]:
        # Some data recording is invalid. Here:
        # - gt_valid: whether the ground truth is valid
        # - aligned: whether the radar recording and GT recording are timestamp-aligned
        # - has_gt: whether the GT recording exists during the experiment
        print(f"Skipping invalid data: {data_dir}")
        return None, None, None, None

    radar = np.load(os.path.join(data_dir, "recording.npy"), allow_pickle=True)
    gt = np.load(os.path.join(data_dir, "respiration_gt.npy"))
    config = np.load(os.path.join(data_dir, "config.npy"), allow_pickle=True).item()
    # calibration = np.load(os.path.join(data_dir, "calibration.npy"))
    gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))  # normalize gt to 0-1
    radar = radar - np.mean(radar[0:5, :, :], axis=0)

    # Get total radar time
    radar_frame_rate = 0.05  # 0.05s per frame
    total_time = metadata["radar_end_time"] - metadata["radar_start_time"]
    gt = gt[
        : int(total_time / radar_frame_rate)
    ]  # sometimes GT recording is longer than radar recording, so truncate gt to radar time

    p = ProcessRadarSlidingWindow(radar, config)
    dist_range = (
        distance - 0.25,
        distance + 0.25,
    )  # e.g. if radar->chest is 0.5m, then only consider radar data within 0.25m to 0.75m range
    doppler = p.process(
        method="range_doppler", freq_range=(0.1, 1.5), dist_range=dist_range
    )

    # Get BPM from gt
    # find both peaks and valleys to increase robustness
    peaks, _ = find_peaks(gt, prominence=0.1, distance=15)
    valleys, _ = find_peaks(-gt, prominence=0.1, distance=15)
    peaks = np.concatenate([peaks, valleys])
    bpm_gt = len(peaks) / total_time * 60 / 2
    plot_peaks(gt, peaks) if plot else None

    # Get BPM from range doppler
    freq, dist = p.get_max_freq(doppler)
    bpm_radar = freq * 60 / 2
    p.plot_max_freq(doppler, freq, dist) if plot else None

    return distance, angle, bpm_gt, bpm_radar


if __name__ == "__main__":
    data_root = Path(__file__).parent.parent / "data"
    distance, angle, bpm_gt, bpm_radar = doppler_baseline(data_root, plot=True)
    print(f"distance: {distance}, angle: {angle}, gt: {bpm_gt}, radar: {bpm_radar}")
