import numpy as np
import os
from scipy.signal import find_peaks
import re
import json

from DopplerProcessing.case import CaseProcessor

def get_ground_truth_respiration(respiration_wave):
    peaks, _ = find_peaks(respiration_wave, prominence=50)
    index_time = np.array([i * 0.05 for i in range(len(respiration_wave))])
    time_intervals = np.diff(index_time[peaks])

    # Calculate the frequencies (in Hz)
    frequencies = 1 / time_intervals

    # Average frequency (rate in Hz)
    # Get average respiration rate
    if len(frequencies) == 0:
        average_frequency = 0.2
    else:
        average_frequency = np.mean(frequencies)
    # print(average_frequency)
    average_BPM = int(average_frequency * 60)
    return average_BPM

def group_scenarios_by_distance(data_path):
    """
    Group the scenarios in the data path by distance.

    Parameters:
    - data_path: The path to the data.

    Returns:
    - A dictionary with the scenarios grouped by the case name.
    """
    scenarios = {}
    for case in os.listdir(data_path):
        case_path = os.path.join(data_path, case)
        # Extract from case name and group
        # e.g. extract 0.5 from "50k_profile_1_distance0.5_degree_0_round_3_Hanbo" using regex
        distance = re.findall(r'distance(\d+\.?\d*)', case)[0]
        scenarios.setdefault(distance, []).append(case)
    return scenarios

if __name__ == "__main__":
    data_path = "0131Aligned"
    scenarios = group_scenarios_by_distance(data_path)
    # print(scenarios)

    result = {}  # distance: {case: {max_freq: , max_freq_dist: }}

    for distance, cases in scenarios.items():
        print(f"Distance: {distance} meters")
        for case in cases:
            print(f"\tProcessing {case}")
            case_path = os.path.join(data_path, case)
            case_processor = CaseProcessor(case_path)
            # respiration_wave = case_processor.get_respiration_wave()
            # respiration_rate = get_ground_truth_respiration(respiration_wave)
            max_freq_list, max_freq_dist_list, index = case_processor.run(plot=False)
            
            result.setdefault(distance, {})[case] = {
                "max_freq": max_freq_list,
                # "max_freq_dist": max_freq_dist_list
                "window_index": index
            }
    
    # Write the result to a file
    with open("result.json", "w") as file:
        json.dump(result, file, indent=4)
        print("Result written to result.json.")