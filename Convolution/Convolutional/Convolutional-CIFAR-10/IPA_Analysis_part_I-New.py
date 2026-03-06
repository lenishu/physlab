import os
import math
import statistics
import numpy as np

# Configurable options
PRUNE_LAYERS_OPTIONS = ['ALL', 'CONV', 'FHL', 'SHL', 'FHL+SHL']
ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(0, 110, 10)]
ACCEPTABLE_BATCH_SIZES = [64, 1024]

def get_run_IPA(path, percentage, bs, run):
    filename = os.path.join(path, f'convol_{percentage}_{bs}_run_{run}.txt')
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

        ce_asymptote = np.mean(ce_values[-20:])
        ce_o = math.log(10)  # ln(10)
        ce_l = ce_o - 0.9 * (ce_o - ce_asymptote)

        learn_batch_number = None
        for i, ce in enumerate(ce_values):
            if ce <= ce_l:
                learn_batch_number = batch_numbers[i]
                ce_learn = ce_values[i]
                if learn_batch_number == 0:
                    print(f"Run {run}: WARNING — learn_batch_number is ZERO at batch index {i}")
                else:
                    print(f"Run {run}: BNL= {learn_batch_number} with CE(BNL) = {ce_learn:.6f} <= CE_l={ce_l:.6f}")
                break

        if learn_batch_number is not None and learn_batch_number > 0:
            ipa = abs(ce_l - ce_o) / learn_batch_number
            return ipa
        else:
            print(f"Run {run}: CE_l not reached or learn_batch_number is 0.")
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

                    for run in range(0, 100):
                        ipa = get_run_IPA(path=directory, percentage=percentage, bs=bs, run=run)
                        if ipa is not None:
                            ipa_values.append(ipa)
                            output_file.write(f'IPA for run {run}: {ipa:.6f}\n')
                        else: # can be removed 
                            output_file.write(f'Run {run}: No valid IPA (skipped)\n') #can be removed

                    if ipa_values:
                        avg_ipa = statistics.mean(ipa_values)
                        std_ipa = statistics.stdev(ipa_values) if len(ipa_values) > 1 else 0.0
                        output_file.write(f"Avg IPA for {directory}: {avg_ipa:.6f}, StdDev IPA: {std_ipa:.6f}\n")
                        print(f"[✓] {directory}: Avg IPA = {avg_ipa:.6f}, StdDev = {std_ipa:.6f}")
                    else:
                        output_file.write("No valid IPA data available\n")
                        print(f"[X] {directory}: No valid IPA data")

if __name__ == "__main__":
    main()

