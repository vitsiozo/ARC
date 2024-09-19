import os
import json

def main():
    # Directory containing the solutions files
    solutions_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/subset50_ARGA_solutions/"

    # Check if the directory exists
    if not os.path.exists(solutions_dir):
        print(f"The directory '{solutions_dir}' does not exist.")
        return

    # List all files in the directory
    files = os.listdir(solutions_dir)

    # Filter for files that match the pattern 'solutions_{task_id}.json'
    solution_files = [f for f in files if f.startswith('solutions_') and f.endswith('.json')]

    if not solution_files:
        print("No solution files found in the directory.")
        return

    # Iterate through each solution file
    for filename in solution_files:
        # Extract task_id from filename
        task_id = filename[len('solutions_'):-len('.json')]

        # Full path to the solution file
        file_path = os.path.join(solutions_dir, filename)

        # Read the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Get the abstraction name
            abstraction_name = data.get('abstraction', 'Unknown')
            print(f"Task {task_id} uses abstraction {abstraction_name}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file '{filename}': {e}")
        except Exception as e:
            print(f"An error occurred while processing file '{filename}': {e}")

if __name__ == "__main__":
    main()
