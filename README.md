### Prerequisites:

- Ubuntu 20.04
- Conda with Python 3.10+
- CUDA >= 11.3 capable GPU
- Python Library in requirements.txt
- Git
- Docker

### Before using Docker/Conda

First clone this repository to local:

```bash
git clone https://github.com/mtang724/BSENSE-in-cabin.git
```

then download data.

### Dataset Preparation

1. Download the test dataset ```test_data.zip``` from the [Link](https://drive.google.com/drive/folders/1IKW6GsTIThGNduqU9UdGtTqJ7JAsQQ93?usp=sharing) and put it under the project base directory. So the current directory looks like:
```bash
user@host:/bsense_main# ls
BSENSE  [other files/folders]  baseline  requirements.txt test_data.zip
```


### Setup with Docker

Before using docker, please ensure the dataset `test_data.zip` is put in the project base directory. 

1. Navigate to the project base directory
2. Build the container with: `docker build -t bsense-in-cabin .` This will
   1. Install all dependencies
   2. Prepare the test dataset
3. Run the container with: `docker run -it --name bsense bsense-in-cabin`

### Steup without Docker

1. Install conda environment with ```conda create -n bsense_env python=3.10```
2. Source environment ```conda activate bsense_env``` and then ```pip install -r requirements.txt```


### Data Preprocessing

To run the minimum working example for data preprocessing:

```bash
cd BSENSE/data_preprocessing
python3 data_preprocessing_for_one_file.py preprocessed_dataset/50k_profile_1_distance1.7_gt_30_round_1
```

The generated feature file will be in ```BSENSE/data_preprocessing/processed_dataset```.

### Evaluation

To run the minimum working example for evaluation on Child Presence Detection and BPM error:

```bash
cd BSENSE/model_training/train_and_inference
python3 indoor_row2_inference.py
```

The above is for backseats front-facing Child Presence Detection - the expected output for 5 experiments:

```
Avg BPM error: 5.2886838734149935
Detection Rate for 5 Experiments: 0.9666666666666667
```

====================================================================================

```bash
cd BSENSE/model_training/train_and_inference
python3 indoor_row3_inference.py
```

The above is for backseats rear-facing Child Presence Detection - the expected output for 5 experiments:

```
Avg BPM error: 6.0980706
Detection Rate for 5 Experiments: 1.0
```

====================================================================================

```bash
cd BSENSE/model_training/train_and_inference
python3 incar_row2_inference.py
```

The above is for baby in car backseats front-facing Child Presence Detection - the expected output for 5 experiments:

```
Avg BPM error: 6.8062005
Detection Rate for 5 Experiments: 0.9222222222222223
```

====================================================================================

Benchmark experiments:

See this [Page](https://github.com/mtang724/BSENSE-in-cabin/tree/main/baseline). It contains both training and testing for BSENSE and baseline signal processing methods. 

=======================================================================================

### Minimum Working Example for Training

```bash
cd BSENSE/model_training/train_and_inference
python3 minimal_example_train.py
```

Expected Output:

```
Epoch 1/60, Training Loss: 611.8685974478722
Epoch 1/60, Validation Loss: 1487.1771697998047
Epoch 2/60, Training Loss: 610.1535022854805
Epoch 2/60, Validation Loss: 1484.2263975143433
Epoch 3/60, Training Loss: 606.7350315451622
Epoch 3/60, Validation Loss: 1486.148323059082
Epoch 4/60, Training Loss: 597.7983624339104
Epoch 4/60, Validation Loss: 1488.603512763977
Epoch 5/60, Training Loss: 577.4779804050922
Epoch 5/60, Validation Loss: 1467.2620077133179
Epoch 6/60, Training Loss: 544.1545549333096
Epoch 6/60, Validation Loss: 1499.2892580032349
Epoch 7/60, Training Loss: 478.4833969473839
Epoch 7/60, Validation Loss: 1042.372676372528
Epoch 8/60, Training Loss: 404.1072451323271
Epoch 8/60, Validation Loss: 1423.732190132141
Epoch 9/60, Training Loss: 302.0514387637377
Epoch 9/60, Validation Loss: 859.2643809318542
Epoch 10/60, Training Loss: 221.59742905199528
Epoch 10/60, Validation Loss: 97.65598768740892
Epoch 11/60, Training Loss: 254.13531486690044
Epoch 11/60, Validation Loss: 65.56008049752563
```



### Dataset Description

Take demo preprocessed_data as an example. We have a ```metadata.json``` file to describe the information and ground truth for each experiments:

```json
{
    "start_freq": 62000.0,
    "stop_freq": 66500.0,
    "sample_time": 0.06611596703529359,
    "RBW": 50,
    "scan_profile": 1,
    "distance": 1.7,
    "round": 1,
    "in_car": false,
    "car_driving": false,
    "real_children": false,
    "where": "CSL",
    "is_benchmark": false,
    "exp_id": 5,
    "exp_comment": "0214/exp5_front_facing_back_right_5m/50k_profile_1_distance1.7_gt_30_round_1",
    "reflector_size": "hemisphere_large",
    "degree": 0,
    "reflector_facing": "front",
    "distance_to_reflector": null,
    "distance_from_reflector_to_chest": null,
    "reflector_comment": "",
    "baby_doll_exists": true,
    "child_doll_exists": false,
    "real_baby_exists": false,
    "real_child_exists": false,
    "real_adult_exists": true,
    "occupied_seats": [
        "driver",
        "passenger",
        "back_left"
    ],
    "seats": {
        "driver": {
            "name": "anonymous_driver",
            "has_gt_device": false,
            "distance_to_radar": null,
            "gt_data_column_name": null,
            "gt": null,
            "front_facing": null
        },
        "passenger": {
            "name": "anonymous_passenger",
            "has_gt_device": false,
            "distance_to_radar": null,
            "gt_data_column_name": null,
            "gt": null,
            "front_facing": null
        },
        "back_left": {
            "name": "baby_doll",
            "has_gt_device": false,
            "distance_to_radar": null,
            "gt_data_column_name": null,
            "gt": 30,
            "front_facing": true
        }
    },
    "radar_start_time": null,
    "radar_end_time": null,
    "gt_start_time": null,
    "gt_end_time": null,
    "aligned": false,
    "alignment_attempted": true,
    "radar_gt_timestamp_matches": null,
    "aligned_radar_start_time": null,
    "aligned_gt_start_time": null,
    "radar_align_index": null,
    "gt_align_index": null,
    "metadata_collected_real_time": false,
    "metadata_collected_after": true,
    "comment": "",
    "collection_date": "2024-02-14",
    "has_recording": true,
    "has_config": true,
    "has_calibration": false,
    "has_gt": false,
    "gt_valid": false
}
```

