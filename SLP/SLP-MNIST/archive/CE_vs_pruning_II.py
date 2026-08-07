# Author: Kevin Jenkins and ChatGPT
# This code uses the CE_test_final_avg_std.txt files generated
# by CE_average_vs_Pruning_percentage.py
# It produces two text files:
# 1) CE_avg_std_per_batch.txt : CE_avg and StdDev for each batch size and pruning %
# 2) CE_best_avg_std.txt : best CE_avg (lowest) among batch sizes for each pruning %

import os
import re

# Function to parse CE_test_final_avg_std.txt
def parse_ce_file(file_path):
    """
    Returns avg CE_test and StdDev CE_test from the CE output file
    """
    avg_ce = None
    std_ce = None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            avg_match = re.search(r'Average Final CE_test:\s+([0-9.eE+-]+)', line)
            std_match = re.search(r'StdDev Final CE_test:\s+([0-9.eE+-]+)', line)
            if avg_match:
                avg_ce = float(avg_match.group(1))
            if std_match:
                std_ce = float(std_match.group(1))
        return avg_ce, std_ce
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None, None

# Function to process directories and generate output
def process_directories():
    root_dir = os.getcwd()
    print("Root directory:", root_dir)

    prune_layers = ["prune_layers_ALL"]
    batch_sizes = [64, 1024, 60000]
    pruning_percentages = [i / 100 for i in range(0, 110, 10)]

    for prune_layer in prune_layers:
        # Output for all batch sizes
        output_file = os.path.join(root_dir, prune_layer, f"{prune_layer}_CE_avg_std_per_batch.txt")
        # Output for best CE (lowest) per pruning %
        output_file_best = os.path.join(root_dir, prune_layer, f"{prune_layer}_CE_best_avg_std.txt")

        with open(output_file, 'w') as out_all, open(output_file_best, 'w') as out_best:
            out_all.write("{:<10} {:<10} {:<20} {:<20}\n".format("P%", "BS", "Avg_CE_test", "StdDev_CE_test"))
            out_best.write("{:<10} {:<10} {:<20} {:<20}\n".format("P%", "BS", "Best_Avg_CE", "StdDev_CE_test"))

            for percentage in pruning_percentages:
                best_ce = float("inf")
                best_std = None
                best_bs = None

                for batch_size in batch_sizes:
                    ce_file_path = os.path.join(
                        root_dir,
                        prune_layer,
                        f"p-percentage_{percentage}",
                        f"batch_size_{batch_size}",
                        "CE_test_final_avg_std.txt"
                    )
                    if os.path.exists(ce_file_path):
                        avg_ce, std_ce = parse_ce_file(ce_file_path)
                        if avg_ce is not None and std_ce is not None:
                            # Write all batch sizes
                            out_all.write("{:<10} {:<10} {:<20.9f} {:<20.9f}\n".format(
                                percentage, batch_size, avg_ce, std_ce
                            ))

                            # Track best CE
                            if avg_ce < best_ce:
                                best_ce = avg_ce
                                best_std = std_ce
                                best_bs = batch_size
                        else:
                            print(f"Error parsing file: {ce_file_path}")
                    else:
                        print(f"File not found: {ce_file_path}")

                # Write best CE per pruning percentage
                if best_bs is not None:
                    out_best.write("{:<10} {:<10} {:<20.9f} {:<20.9f}\n".format(
                        percentage, best_bs, best_ce, best_std
                    ))
                else:
                    out_best.write("{:<10} {:<10} {:<20} {:<20}\n".format(
                        percentage, "N/A", "N/A", "N/A"
                    ))


def main():
    process_directories()

if __name__ == "__main__":
    main()

