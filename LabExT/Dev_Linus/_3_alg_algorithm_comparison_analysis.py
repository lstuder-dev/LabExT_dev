import csv
import numpy as np
from pathlib import Path
import statistics

def load_results(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))

def summarize_by_method(my_data_list):
    """Average best_power and n_samples per method, across all files/seeds."""

    def _filter(side, method):
        return [row for row in my_data_list if row['side'] == side and row['method'] == method]

    coordinate_methods = ['coordinate_gaussian', 'coordinate_measured', 'coordinate_quadratic']
    other_methods = ['bayesian', 'bisection', 'golden_section', 'grid_search', 'fixed_gradient_ascent']

    all_lists = {}
    for side in ('left', 'right'):
        for method in coordinate_methods:
            all_lists[f"{side}_{method}"] = _filter(side, method)
            all_lists[f"{side}_{method}_dual"] = _filter(side, f"{method}_dual")
        for method in other_methods:
            all_lists[f"{side}_{method}"] = _filter(side, method)

    
    # Loop through each list and calculate the metrics
    results = {}

    for list_name, data_list in all_lists.items():
        if not data_list:
            continue

        scores = []
        samples = []
        for row in data_list:
            try:
                scores.append(float(row["best_power"]))
            except (ValueError, TypeError):
                continue

            try:
                samples.append(float(row["n_samples"]))
            except (ValueError, TypeError):
                pass

        if not scores:
            continue

        results[list_name] = {
            "count": len(scores),
            "avg": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else None,
            "max": max(scores),
            "min": min(scores),
            "avg_samples": statistics.mean(samples) if samples else None,
        }

    LABEL_WIDTH = 33

    # Helper function to print a specific metric sorted highest -> lowest per side
    def print_sorted_metric(title, metric_key):
        print(f"=== {title.upper()} ===")
        
        for side in ('left', 'right'):
            # Filter items for the current side
            side_items = [
                (name, stats) for name, stats in results.items() 
                if name.startswith(f"{side}_")
            ]
            
            # Sort items highest to lowest based on the specified metric
            # Handles None values (e.g., missing std dev) by placing them at the end (-inf)
            sorted_items = sorted(
                side_items, 
                key=lambda item: item[1][metric_key] if item[1][metric_key] is not None else float('-inf'), 
                reverse=True
            )
            
            print(f"\n  --- {side.upper()} SIDE ---")
            for name, stats in sorted_items:
                val = stats[metric_key]
                if val is not None:
                    print(f"    {name:<{LABEL_WIDTH}} : {val:.6f}")
                else:
                    print(f"    {name:<{LABEL_WIDTH}} : N/A (only 1 data point)")
        print()

    # Print sections sorted highest to lowest
    print_sorted_metric("Averages", "avg")
    print_sorted_metric("Standard Deviations", "std")
    print_sorted_metric("Average N Samples", "avg_samples")
    print_sorted_metric("Highest Values", "max")
    print_sorted_metric("Lowest Values", "min")

rows = load_results(Path(__file__).resolve().parent / "_3_algorithm_comparison_results.csv")
summarize_by_method(rows)