
from pathlib import Path
import pandas as pd


def csv_to_arrays(file_name):
    program_folder = Path(__file__).resolve().parent
    csv_file = program_folder / f"{file_name}.csv"

    df = pd.read_csv(csv_file)

    numeric_columns = [
        "time_taken_s",
        "start_left_x_um",
        "start_left_y_um",
        "start_right_x_um",
        "start_right_y_um",
        "end_left_x_um",
        "end_left_y_um",
        "end_right_x_um",
        "end_right_y_um",
        "ending_power_dBm",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    results = {}

    for algorithm, group in df.groupby("algorithm"):

        group = group.copy()

        best_idx = group["ending_power_dBm"].idxmax()
        best = group.loc[best_idx]

        max_power = group["ending_power_dBm"].max()
        min_power = group["ending_power_dBm"].min()

        max_x_l = best["end_left_x_um"]
        max_y_l = best["end_left_y_um"]

        max_x_r = best["end_right_x_um"]
        max_y_r = best["end_right_y_um"]

        valid_pos_l = group.dropna(subset=["start_left_x_um","start_left_y_um","end_left_x_um", "end_left_y_um"])
        valid_pos_r = group.dropna(subset=["start_right_x_um","start_right_y_um","end_right_x_um", "end_right_y_um"])

        distances_l = (
            (valid_pos_l["end_left_x_um"] - valid_pos_l["start_left_x_um"]) ** 2
            + (valid_pos_l["end_left_y_um"] - valid_pos_l["start_left_y_um"]) ** 2
        ) ** 0.5

        distances_r = (
            (valid_pos_r["end_right_x_um"] - valid_pos_r["start_right_x_um"]) ** 2
                + (valid_pos_r["end_right_y_um"] - valid_pos_r["start_right_y_um"]) ** 2
        ) ** 0.5
        
        results[algorithm] = {
            "average_measured_dBm":
                group["ending_power_dBm"].mean(),

            "std_measured_dBm":
                group["ending_power_dBm"].std(ddof=0),

            "average_x_left_um": valid_pos_l["end_left_x_um"].mean(),
            "std_x_left_um": valid_pos_l["end_left_x_um"].std(ddof=0),

            "average_y_left_um": valid_pos_l["end_left_y_um"].mean(),
            "std_y_left_um": valid_pos_l["end_left_y_um"].std(ddof=0),

            "average_x_right_um": valid_pos_r["end_right_x_um"].mean(),
            "std_x_right_um": valid_pos_r["end_right_x_um"].std(ddof=0),

            "average_y_right_um": valid_pos_r["end_right_y_um"].mean(),
            "std_y_right_um": valid_pos_r["end_right_y_um"].std(ddof=0),

            "average_distance_travelled_left_um": distances_l.mean(),
            "std_distance_travelled_left_um": distances_l.std(ddof=0),

            "average_distance_travelled_right_um": distances_r.mean(),
            "std_distance_travelled_right_um": distances_r.std(ddof=0),

            "average_time_s": group["time_taken_s"].mean(),
        }

    return results


if __name__ == "__main__":
    results = csv_to_arrays("ID1_peak_search_results")

    for algorithm, stats in results.items():
        print(f"\n--- {algorithm} ---")

        for name, value in stats.items():
            if pd.isna(value):
                print(f"{name}: N/A")
            else:
                print(f"{name}: {value:.4f}")
