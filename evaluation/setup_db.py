import os
from pathlib import Path
from data_construction.data.data_manager import MTDataManager

# Get the directory of the current file
current_dir = Path(__file__).parent
# Get the path to the dataset directory (relative to the parent of the current file)
dataset_dir = current_dir.parent / "dataset"

# Four dataset files
dataset_files = [
    "java_function.jsonl",
    "java_repository.jsonl",
    "python_function.jsonl",
    "python_repository.jsonl"
]

# Create a temporary merged file
merged_file = current_dir / "merged_dataset.jsonl"

# Merge all dataset files into one file
with open(merged_file, 'w', encoding='utf-8') as outfile:
    for dataset_file in dataset_files:
        dataset_path = dataset_dir / dataset_file
        print(f"Reading: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():  # Skip empty lines
                    outfile.write(line)

print(f"All datasets merged into: {merged_file}")

# Initialize database
manager = MTDataManager()
print("Initializing database...")
manager.setup_with_jsonl(merged_file)
print(f"Database initialized! Imported {manager.count()} records")

# Remove temporary merged file
if merged_file.exists():
    os.remove(merged_file)
    print(f"Deleted temporary file: {merged_file}")
