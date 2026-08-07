
#Author: Kevin Jenkins
#This code calculates the IPA for each run for each batch size for each pruning percentage for each pruning layer. 
#A single file is created for each batch size directory. The file contains the IPA for each run and at the bottom lists the IPA_avg and STD for the runs in that directory.
#A separate code finds the highest IPA_avg among the different batch_sizes. That is the IPA that will be plotted for that pruning percentage and pruning layer combo.
#Line 45 can be modified depending on the number of runs available.


import os
import statistics

PRUNE_LAYERS_OPTIONS = ['CONV', 'FHL', 'SHL', 'FHL+SHL', 'ALL']
ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(0, 110, 10)]
ACCEPTABLE_BATCH_SIZES = [64,60000]

def get_run_IPA(path, percentage, bs, run):
    # Construct full file path
    filename = os.path.join(path, f'convol_{percentage}_{bs}_run_{run}.txt')
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract relevant values
        first_ce_test = float(lines[1].split()[4])  # CE_TEST from the second line
        last_ce_test = float(lines[-1].split()[4])  # CE_TEST from the last line
        final_batch_number = int(lines[-1].split()[5])  # Batch_Number from the last line

        # Calculate IPA
        result = (first_ce_test - last_ce_test) / final_batch_number
        return result
    except Exception as e:
        return None  # Return None to indicate failure

def main():
    for layer in PRUNE_LAYERS_OPTIONS:
        for percentage in ACCEPTABLE_PRUNE_PERCENTAGES:
            for bs in ACCEPTABLE_BATCH_SIZES:
                directory = os.path.join(f'prune_layers_{layer}', f'p-percentage_{percentage}', f'batch_size_{bs}')
                output_filename = os.path.join(directory, 'IPA_avg_std.txt')
                os.makedirs(directory, exist_ok=True)  # Ensure directory exists
                
                with open(output_filename, 'w') as output_file:
                    ipa_values = []

                    output_file.write(f'Processing path: {directory}\n')
                    for run in range(41):
                        ipa = get_run_IPA(path=directory, percentage=percentage, bs=bs, run=run)
                        if ipa is not None:
                            ipa_values.append(ipa)
                            output_file.write(f'IPA for run {run}: {ipa:.6f}\n')

                    if ipa_values:
                        average_ipa = statistics.mean(ipa_values)
                        stddev_ipa = statistics.stdev(ipa_values) if len(ipa_values) > 1 else 0
                        output_file.write(f"Avg IPA for {directory}: {average_ipa:.6f}, StdDev IPA: {stddev_ipa:.6f}\n")
                    else:
                        output_file.write("No valid IPA data available\n")

if __name__ == "__main__":
    main()

