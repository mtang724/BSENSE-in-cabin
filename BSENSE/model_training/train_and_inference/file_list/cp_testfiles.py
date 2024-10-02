import shutil
import os

# Original prefix to replace
old_prefix = '/projects/bchl/mt55/sep_processed_data/'

# New prefix to use in place of the old one
new_prefix = '../test_data'

for i in range(0, 6):
    try:
        # Path to the text file containing the list of file paths
        folder_list_path = '/u/mt55/BSENSE/model_training/train_and_inference/file_list/indoor_row3_file_list/test_filenames_fold_{}.txt'.format(i)

        # Destination folder where files will be copied
        destination_folder = '/u/mt55/BSENSE/model_training/test_data'

        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

        # Read the folder paths from the text file and copy them
        with open(folder_list_path, 'r') as file:
            for line in file:
                folder_path = line.strip()
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    folder_name = os.path.basename(folder_path)
                    dest_path = os.path.join(destination_folder, folder_name)
                    shutil.copytree(folder_path, dest_path)
                    print(f"Copied: {folder_path}")
                else:
                    print(f"Folder not found or not a directory: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        continue
    
import os

# Folder whose file paths you want to list
source_folder = '/u/mt55/BSENSE/model_training/test_data'

# Path to the text file where file paths will be written
output_txt = '/u/mt55/BSENSE/model_training/train_and_inference/file_list/experiment_list/text_file_list.txt'

# Open the text file in write mode
with open(output_txt, 'w') as file:
    # Walk through the directory
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # Get the full file path
            file_path = os.path.join(root, filename)
            # Write the file path to the text file
            file.write(file_path + '\n')

print(f"File paths have been written to {output_txt}")