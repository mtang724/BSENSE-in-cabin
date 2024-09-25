# Extracts metadata from each data point retrospectively (after doing the exp)
import json
import os


class Colors:
    """Color class for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Seats:
    """Class to conveniently add seat information to the metadata."""
    def __init__(self):
        self.seats = {}

    def add_seat(self, which_seat, name, has_gt_device, gt=None, front_facing=None, distance_to_radar=None):
        self.seats[which_seat] = {
            "name": name,
            "has_gt_device": has_gt_device,
            "distance_to_radar": distance_to_radar,
            "gt_data_column_name": which_seat if has_gt_device else None,
            "gt": gt,
            "front_facing": front_facing
        }
    
    def to_dict(self):
        return self.seats
    
    def __setitem__(self, key, value):
        self.seats[key] = value
    
    def __getitem__(self, key):
        return self.seats[key]

    def __iter__(self):
        return iter(self.seats)
    

class MetadataBase:
    def __init__(self, name):
        """Initialize the metadata with the given name, e.g. "metadata.json"."""
        self.metadata = {}
        self.name = name
                
    def fill(self, **kwargs):
        """
        Warning: None values are ignored.
        """
        for key, value in kwargs.items():
            if key in self.metadata and value is not None:
                self.metadata[key] = value
    
    def save(self, metadata_path, check_on_update=True):
        """Save the metadata to the given path (path only, exclude file name). 
        
        If check_on_update is True, it will check if the metadata is updated compared to the existing one.
        """
        if check_on_update:
            self._check_update(metadata_path)
        with open(os.path.join(metadata_path, self.name), "w") as f:
            json.dump(self.metadata, f, indent=4)

    def read(self, metadata_path):
        """Read the metadata from the given path (path only, exclude file name)."""
        with open(os.path.join(metadata_path, self.name), "r") as f:
            self.metadata = json.load(f)
        return self

    def remove_field(self, key):
        if key in self.metadata:
            del self.metadata[key]

    def to_dict(self):
        return self.metadata
    
    def _check_update(self, metadata_path):
        """Check if the metadata is updated compared to older version, prints the differences."""
        metadata_path = os.path.join(metadata_path, self.name)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
            if existing_metadata != self.metadata:
                print(f"{Colors.WARNING}Metadata updated detected: {Colors.OKBLUE}{metadata_path}{Colors.ENDC}")
                # Find what is updated
                for key, value in self.metadata.items():
                    if key not in existing_metadata:
                        print(f"{Colors.OKCYAN}\t{key}:{Colors.ENDC} _ {Colors.HEADER}-> {Colors.OKGREEN}{value}{Colors.ENDC}")
                    elif existing_metadata[key] != value:
                        if key == "seats":
                            self._print_seat_update(existing_metadata[key], value)
                        else:
                            print(f"{Colors.OKCYAN}\t{key}:{Colors.ENDC} {existing_metadata[key]} {Colors.HEADER}-> {Colors.OKGREEN}{value}{Colors.ENDC}")
                # Find what is deleted
                for key in existing_metadata:
                    if key not in self.metadata:
                        print(f"{Colors.OKCYAN}\t{key}:{Colors.ENDC} {existing_metadata[key]} {Colors.HEADER}-> {Colors.FAIL}DELETE{Colors.ENDC}")

                # key "seats" is like {'driver': {'name': 'Hanbo', 'has_gt_device': True, 'distance_to_radar': None, 'gt_data_column_name': 'driver', 'gt': None, 'front_facing': None}, 'back_right': ...}
                # detect what fields is updated in seats
                input("Press Enter to continue...")

    def _print_seat_update(self, existing_seat_info, new_seat_info):
        """Print the differences between the existing and new seat information."""
        for seat, seat_info in new_seat_info.items():
            if seat not in existing_seat_info:
                print(f"{Colors.OKCYAN}\tseats: {seat}:{Colors.ENDC} _ {Colors.HEADER}-> {Colors.OKGREEN}{seat_info}{Colors.ENDC}")
            else:
                for key, value in seat_info.items():
                    if key not in existing_seat_info[seat]:
                        print(f"{Colors.OKCYAN}\tseats: {seat}:{Colors.ENDC} _ {Colors.HEADER}-> {Colors.OKGREEN}{key}:{value}{Colors.ENDC}")
                    elif existing_seat_info[seat][key] != value:
                        print(f"{Colors.OKCYAN}\tseats: {seat}:{Colors.ENDC} {key}:{existing_seat_info[seat][key]} {Colors.HEADER}-> {Colors.OKGREEN}{value}{Colors.ENDC}")

    def __getitem__(self, key):
        return self.metadata[key]

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __repr__(self):
        # Pretty print json
        return json.dumps(self.metadata, indent=4)


class MetadataManager(MetadataBase):
    """Metadata manager for each data point."""

    REQUIRED_FIELDS = ["distance", "round", "where"]

    def __init__(self, name):
        super().__init__(name)

        # Defines the metadata fields
        self.metadata = {
            # Radar setup
            "start_freq": None,
            "stop_freq": None,
            "sample_time": None,

            # Experiment setup
            "RBW": 50,  # Range-bin-window of the radar sensor
            "scan_profile": 1,  # The scan profile used by the radar sensor
            "distance": None,  # Distance from the radar to the moving object (in meters)
            "round": None,  # The round of the experiment
            
            "in_car": None,  # Whether the experiment was conducted inside a car
            "car_driving": False,  # Whether the car was driving during the experiment
            "real_children": False,  # Whether a human children was involved in the experiment
            "where": None,  # The location where the experiment was conducted (e.g., CSL or BMW)
            "is_benchmark": False,  # Whether the experiment is a benchmark
            "exp_id": None,  # The ID or number assigned to the experiment, according to the experiment plan
            "exp_comment": "",  # The description of the specific experimental case, taken from the original folder names

            # Reflector setup
            "reflector_size": "hemisphere_large",  # The type of reflector used in the experiment (e.g., "large", "small")
            "degree": 0,  # Degree of the reflector relative to the object
            "reflector_facing": "front",  # The direction the reflector is facing (e.g., "front", "back")
            "distance_to_reflector": None,  # The distance between the radar and the reflector
            "distance_from_reflector_to_chest": None,  # The distance between the reflector and the chest of the object
            "reflector_comment": "",  # Any other information for the reflector, explaining something

            # Subjects
            "baby_doll_exists": False,  # Whether a baby doll was present
            "child_doll_exists": False,  # Whether a child doll was present
            "real_baby_exists": False,  # Whether a real baby was present
            "real_child_exists": False,  # Whether a real child was present
            "real_adult_exists": False,  # Whether a real adult was present

            # Seats information
            "occupied_seats": [],  # The seats that were occupied (e.g., ["driver", "passenger", "back_left", "back_right", "back_middle"])
            "seats": {},  # Dictionary containing information about each occupied seat

            # Data collection times
            "radar_start_time": None,  # The start time of the radar data collection
            "radar_end_time": None,  # The end time of the radar data collection
            "gt_start_time": None,  # The start time of the groud truth data collection
            "gt_end_time": None,  # The end time of the groud truth data collection

            # Alignment information
            "aligned": False,  # Whether the radar and seatbelt data are aligned
            "alignment_attempted": False,  # Whether we tried to align the radar and ground truth data
            "radar_gt_timestamp_matches": None,  # Whether the radar and ground truth data timestamps match
            "aligned_radar_start_time": None,  # The start time of the aligned radar data
            "aligned_gt_start_time": None,  # The start time of the aligned ground truth data
            "radar_align_index": None,  # Radar data aligns with GT data starting from this index
            "gt_align_index": None,  # GT data aligns with radar data starting from this index

            # Metadata collection
            "metadata_collected_real_time": False,  # Whether the metadata was collected in real-time while collecting data
            "metadata_collected_after": False,  # Whether the metadata was analyzed after the experiment retrospectively
        
            # General fields
            "comment": "",  # Any other comments
            "collection_date": None,  # The date when the data was collected
            "has_recording": True,  # If data point has `recording.npy`, the radar data file.
            "has_config": True,  # If data point has `config.npy`, the radar config file.
            "has_calibration": False, # If data point has `calibration.npy`, the radar background calibration file.
            "has_gt": True,  # If data point has `respiration_belt.csv`, the groun truth file.

            "gt_valid": False,  # Whether the ground truth data is valid (has gt and gt columns valid)
        }
            
    def fill(self, **kwargs):
        """
        Fill in the metadata dictionary with the provided key-value pairs.
        Raises a ValueError if any required field is missing.

        """
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in kwargs]
        if missing_fields:
            raise ValueError(f"The following required fields are missing: {', '.join(missing_fields)}")

        super().fill(**kwargs)

        # Extra rules to fill in the metadata
        if self.metadata["where"] == "CSL":
            self.metadata["in_car"] = False


class GroundTruthStatistics(MetadataBase):
    """Stores statistics of ground truth data."""
    
    def __init__(self, name):
        super().__init__(name)

        self.metadata = {
            "gt_path": None,
            "gt_window_size": None,
            "gt_window_stride": None,
            "gt_window_bpm": None,
        }
