import json

# Define the color mapping dictionary
color_mapping = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "fuchsia",
    7: "orange",
    8: "teal",
    9: "brown"
}

# Function to replace numbers with corresponding color names in a challenges file
def replace_with_colors_challenge(data, color_mapping):
    for key, value in data.items():
        for section in ["test", "train"]:
            if section in value:
                for example in value[section]:
                    if "input" in example:
                        example["input"] = [[color_mapping[val] for val in row] for row in example["input"]]
                    if "output" in example:
                        example["output"] = [[color_mapping[val] for val in row] for row in example["output"]]
    return data

# Function to replace numbers with corresponding color names in a solutions file
def replace_with_colors_solutions(data, color_mapping):
    for key, value in data.items():
        # Each key corresponds to a list of lists (2D array), so we iterate over those
        data[key] = [[[color_mapping[val] for val in row] for row in sublist] for sublist in value]
    return data

# Function to load a JSON file, transform it, and save the result to a new file
def process_json(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    transformed_data = replace_with_colors_solutions(data, color_mapping)
    
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile)

# Example usage: replace 'input.json' with your actual input file and 'output.json' with the desired output file name.
input_file = '50_solutions.json'  # Replace with your input file
output_file = '50_word_in_quotes_solutions.json'  # Replace with your output file
process_json(input_file, output_file)
