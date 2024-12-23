import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import scipy.stats as stats

def parse_log(file_path):
    durations = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "TIME SPENT" in line:
                duration = float(line.split("TIME SPENT:")[1].strip().split(" ")[0])
                durations.append(duration)
    return durations

def save_results_to_file(results, output_path):
    with open(output_path, "w") as file:
        file.write("Statistical Results:\n")
        file.write(f"Mean for {results['file1_name']}: {results['mean_1']:.2f} minutes, standard deviation: {results['std_1']:.2f}\n")
        file.write(f"Mean for {results['file2_name']}: {results['mean_2']:.2f} minutes, standard deviation: {results['std_2']:.2f}\n")
        file.write(f"{results['file2_name']} is {'{:.2f}'.format(results['percentage_diff'])}% smaller than {results['file1_name']}.\n")
        file.write(f"t-statistic: {results['t_stat']:.4f}, p-value: {results['p_value']:.4f}\n")
        if results["significant"]:
            file.write(f"The results are statistically significant at the alpha level of {results['alpha']}.\n")
        else:
            file.write(f"The results are not statistically significant at the alpha level of {results['alpha']}.\n")

def plot_t_distribution(results, output_path):
    df = len(results["durations_1"]) + len(results["durations_2"]) - 2
    critical_value = t.ppf(1 - results['alpha']/2, df)
    x = np.linspace(-5, 5, 500)
    y = t.pdf(x, df)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="t-distribution", color="blue")
    plt.axvline(results["t_stat"], color="red", linestyle="--", label=f"Observed t ({results['t_stat']:.2f})")
    plt.axvline(critical_value, color="green", linestyle="--", label=f"Critical value ±{critical_value:.2f}")
    plt.axvline(-critical_value, color="green", linestyle="--")
    plt.fill_between(x, 0, y, where=(x >= critical_value) | (x <= -critical_value), color="green", alpha=0.3, label="Critical region")
    plt.title("t-distribution and critical regions")
    plt.xlabel("t-value")
    plt.ylabel("Probability density")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def compare_and_save(file1, file2, output_dir, alpha=0.05):
    durations1 = parse_log(file1)
    durations2 = parse_log(file2)

    t_stat, p_value = stats.ttest_ind(durations1, durations2, equal_var=False)

    mean_1 = np.mean(durations1)
    mean_2 = np.mean(durations2)
    percentage_diff = ((mean_1 - mean_2) / mean_1) * 100

    results = {
        "file1_name": os.path.splitext(os.path.basename(file1))[0],
        "file2_name": os.path.splitext(os.path.basename(file2))[0],
        "mean_1": mean_1,
        "std_1": np.std(durations1, ddof=1),
        "mean_2": mean_2,
        "std_2": np.std(durations2, ddof=1),
        "percentage_diff": percentage_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "durations_1": durations1,
        "durations_2": durations2,
    }

    result_file = os.path.join(output_dir, "statistical_results.txt")
    save_results_to_file(results, result_file)

    plot_file = os.path.join(output_dir, "t_distribution_plot.png")
    plot_t_distribution(results, plot_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two log files and analyze statistical significance.")
    parser.add_argument("--file1", required=True, help="Path to the first log file.")
    parser.add_argument("--file2", required=True, help="Path to the second log file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the results and plot.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    compare_and_save(args.file1, args.file2, args.output_dir, args.alpha)
