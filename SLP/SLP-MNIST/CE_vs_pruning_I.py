import os
import statistics
import numpy as np

# Configurable options
PRUNE_LAYERS_OPTIONS = ['ALL']
ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(0, 110, 10)]
ACCEPTABLE_BATCH_SIZES = [64, 1024, 60000]
NUM_RUNS = 100


def get_final_CE_test(path, percentage, bs, run):
    """
    Reads the final CE_test value from a run file.
    Assumes CE_test is column index 4 (same as original code).
    """
    filename = os.path.join(path, f'slp_{percentage}_{bs}_run_{run}.txt')

    try:
        print(f"Opening file: {filename}")
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Remove header
        lines = lines[1:]

        if not lines:
            print(f"Run {run}: File is empty after header")
            return None

        # Take CE_test from the last line
        final_line = lines[-1]
        final_ce = float(final_line.split()[4])

        return final_ce

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def main():
    for layer in PRUNE_LAYERS_OPTIONS:
        for percentage in ACCEPTABLE_PRUNE_PERCENTAGES:
            for bs in ACCEPTABLE_BATCH_SIZES:
                directory = os.path.join(
                    f'prune_layers_{layer}',
                    f'p-percentage_{percentage}',
                    f'batch_size_{bs}'
                )

                os.makedirs(directory, exist_ok=True)

                output_filename = os.path.join(
                    directory,
                    'CE_test_final_avg_std.txt'
                )

                final_ce_values = []

                with open(output_filename, 'w') as output_file:
                    output_file.write(f'Processing path: {directory}\n')

                    for run in range(NUM_RUNS):
                        final_ce = get_final_CE_test(
                            path=directory,
                            percentage=percentage,
                            bs=bs,
                            run=run
                        )

                        if final_ce is not None:
                            final_ce_values.append(final_ce)
                            output_file.write(
                                f'Final CE_test for run {run}: {final_ce:.6f}\n'
                            )

                    if final_ce_values:
                        avg_ce = statistics.mean(final_ce_values)
                        std_ce = (
                            statistics.stdev(final_ce_values)
                            if len(final_ce_values) > 1 else 0.0
                        )

                        output_file.write(
                            f'\nAverage Final CE_test: {avg_ce:.6f}\n'
                        )
                        output_file.write(
                            f'StdDev Final CE_test: {std_ce:.6f}\n'
                        )
                    else:
                        output_file.write(
                            '\nNo valid CE_test data available\n'
                        )


if __name__ == "__main__":
    main()

