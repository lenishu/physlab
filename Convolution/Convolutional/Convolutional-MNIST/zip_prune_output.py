import os
import zipfile

def find_output_files(root_dir):
    target_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("_output.txt") or filename.endswith("_output_highest.txt"):
                if filename.startswith("prune_layers"):
                    full_path = os.path.join(dirpath, filename)
                    target_files.append(full_path)

    return target_files

def zip_files(file_paths, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in file_paths:
            arcname = os.path.relpath(file, start=os.getcwd())
            zipf.write(file, arcname)
    print(f"\nCreated ZIP archive: {zip_filename}")

def main():
    root_dir = os.getcwd()
    print("Searching from root:", root_dir)

    found_files = find_output_files(root_dir)

    if not found_files:
        print("No matching output files found.")
        return

    print(f"\nFound {len(found_files)} output files:")
    for f in found_files:
        print(f)

    zip_filename = "prune_output_files.zip"
    zip_files(found_files, zip_filename)

if __name__ == "__main__":
    main()

