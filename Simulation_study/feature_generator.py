##################################################### #
# C. Jarne 2024 Analysis Group of D. Vidahurre  @cfin #
# To generate the features files used in prediction   #
# for syntetic data                                   #
#######################################################
import os
import re

# Input directory
input_directory = "synthetic-data-grid"
# Output directory
output_directory = "features_syntetic"

# Regular expression to find the label and age type in the file name
pattern = re.compile(r'sample_(\d+)_(\w+).txt')

# Iterate over directories in the input directory
for directory_name in os.listdir(input_directory):
    directory_path = os.path.join(input_directory, directory_name)
    # Check if the element is a directory
    if os.path.isdir(directory_path):
        # Dictionary to store labels and their corresponding values
        labels_dict = {}
        # Iterate over files in the current directory
        for file_name in os.listdir(directory_path):
            # Check if the file is a text file
            if file_name.endswith(".txt"):
                # Look for matches with the pattern in the file name
                match = pattern.match(file_name)
                if match:
                    # Get the number and age type from the file name
                    sample_number = match.group(1)
                    age_type = match.group(2)
                    # Assign a numeric value to the age (1 for old, -1 for young)
                    age_value = 1 if age_type == "old" else -1
                    # Add the label and value to the dictionary
                    label = f"sample_{sample_number}_{age_type}"
                    labels_dict[label] = age_value
        # Output file name with the directory name
        output_file = os.path.join(f"features_syntetic_{directory_name}.txt")
        # Write labels and values to the output file
        with open(output_file, "w") as f:
            f.write("Label\tValue\n")
            for label, value in labels_dict.items():
                f.write(f"{label}\t{value}\n")

print("Files successfully generated in the 'features_syntetic' directory.")
