import os
import math
import statistics
import numpy as np

# Configurable options
PRUNE_LAYERS_OPTIONS = ['ALL']
ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(0, 110, 10)]
ACCEPTABLE_BATCH_SIZES = [64,1024, 60000]

def get_run_IPA(path, percentage, bs, run):
    filename = os.path.join(path, f'slp_{percentage}_{bs}_run_{run}.txt')
    try:
        print(f"Opening file: {filename}")
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Skip header
        lines = lines[1:]

        # Extract CE and batch numbers
        ce_values = [float(line.split()[4]) for line in lines]
        batch_numbers = [int(line.split()[5]) for line in lines]

        if len(ce_values) < 20:
            print(f"Run {run}: Not enough CE values to compute CE_asymptote")
            return None

        # Compute CE_asymptote as the mean of the last 20 CE values
        ce_asymptote = np.mean(ce_values[-20:])
        ce_o = math.log(10)  # ln(10)
        ce_l = ce_o - 0.9 * (ce_o - ce_asymptote)

        # Find the first batch number where CE <= CE_l
        learn_batch_number = None
        for i, ce in enumerate(ce_values):
            if ce <= ce_l:
                learn_batch_number = batch_numbers[i]
                ce_learn = ce_values[i]
                print(f"Run {run}: BNL= {learn_batch_number} with CE(BNL) = {ce_learn:.6f} less than CE_l={ce_l:.6f}")
                break

        # Compute IPA if batch found
        if learn_batch_number is not None:
            ipa = abs(ce_l - ce_o) / (learn_batch_number)
            return ipa
        else:
            print(f"Run {run}: CE_l not reached in data")
            return None

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main():
    for layer in PRUNE_LAYERS_OPTIONS:
        for percentage in ACCEPTABLE_PRUNE_PERCENTAGES:
            for bs in ACCEPTABLE_BATCH_SIZES:
                directory = os.path.join(f'prune_layers_{layer}', f'p-percentage_{percentage}', f'batch_size_{bs}')
                os.makedirs(directory, exist_ok=True)
                output_filename = os.path.join(directory, 'IPA_avg_std.txt')

                with open(output_filename, 'w') as output_file:
                    ipa_values = []
                    output_file.write(f'Processing path: {directory}\n')

                    for run in range(0,100):
                        ipa = get_run_IPA(path=directory, percentage=percentage, bs=bs, run=run)
                        if ipa is not None:
                            ipa_values.append(ipa)
                            output_file.write(f'IPA for run {run}: {ipa:.6f}\n')

                    if ipa_values:
                        avg_ipa = statistics.mean(ipa_values)
                        std_ipa = statistics.stdev(ipa_values) if len(ipa_values) > 1 else 0.0
                        output_file.write(f"Avg IPA for {directory}: {avg_ipa:.6f}, StdDev IPA: {std_ipa:.6f}\n")
                    else:
                        output_file.write("No valid IPA data available\n")

if __name__ == "__main__":
    main()
