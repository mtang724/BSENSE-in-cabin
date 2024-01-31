import requests
import json
import time

# Define the API URLs
start_exp_url = "http://localhost:22004/NeuLogAPI?StartExperiment:[Respiration],[1],[7],[600]"
end_exp_url = "http://localhost:22004/NeuLogAPI?StopExperiment"
get_data_url = "http://localhost:22004/NeuLogAPI?GetExperimentSamples"

round = 1
distance1 = 0.8 # in meters
# distance2 = 0.6
degree = 0 # in degrees

time.sleep(5)
# Send a request to start the experiment
response_start = requests.get(start_exp_url)
start_data = response_start.json()  # Assuming it returns JSON

# Check if the experiment was successfully started
if "StartExperiment" in start_data and start_data["StartExperiment"] == "True":
    start_timestamp = time.time()
    print("Experiment started successfully.")
else:
    print("Failed to start the experiment.")
    exit()
sample_size = 600
timestamp_list = [start_timestamp]
for i in range(sample_size):
    timestamp_list.append(timestamp_list[-1] + 0.05)
time.sleep(31)
# Send a request to get the experiment samples
response_data = requests.get(get_data_url)
data = response_data.json()  # Assuming it returns JSON

# Check if the data retrieval was successful
if "GetExperimentSamples" in data:
    # Extract the data points from the response
    samples = data["GetExperimentSamples"][0]
    # Save the data to a file (e.g., a CSV file)
    with open("{}_front_{}_{}.csv".format(round, distance1, degree), "w") as file:
        for timestamp, sample in zip(timestamp_list, samples[2:]):
            file.write(str(timestamp) + "," + str(sample) + "\n")
            # Add 1/20th of a second (0.05 seconds) to the timestamp
            # timestamp += 0.05

    print("Data saved to experiment_data.csv.")
else:
    print("Failed to retrieve experiment data.")
    
# Send a request to stop the experiment
response_end = requests.get(end_exp_url)
end_data = response_end.json()  # Assuming it returns JSON

# Check if the experiment was successfully stopped
if "StopExperiment" in end_data and end_data["StopExperiment"] == "True":
    print("Experiment stopped successfully.")
else:
    print("Failed to stop the experiment.")
    exit()