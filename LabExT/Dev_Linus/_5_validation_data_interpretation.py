
from pathlib import Path
import pandas as pd


def csv_to_arrays(file_name, measured_max):
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

    df["ending_power_mW"] = 10 ** (df["ending_power_dBm"] / 10)

    results = {}

    for algorithm, group in df.groupby("algorithm"):

        group = group.copy()

        best_idx = group["ending_power_mW"].idxmax()
        best = group.loc[best_idx]

        max_power = group["ending_power_mW"].max()
        min_power = group["ending_power_mW"].min()

        max_x_l = best["end_left_x_um"]
        max_y_l = best["end_left_y_um"]

        max_x_r = best["end_right_x_um"]
        max_y_r = best["end_right_y_um"]

        valid_pos_l = group.dropna(subset=["start_left_x_um","start_left_y_um","end_left_x_um", "end_left_y_um"])
        valid_pos_r = group.dropna(subset=["start_right_x_um","start_right_y_um","end_right_x_um", "end_right_y_um"])

        std_lx = valid_pos_l["end_left_x_um"].std(ddof=0)
        std_ly = valid_pos_l["end_left_y_um"].std(ddof=0)
        std_rx = valid_pos_r["end_right_x_um"].std(ddof=0)
        std_ry = valid_pos_r["end_right_y_um"].std(ddof=0)

        distances_l = (
            (valid_pos_l["end_left_x_um"] - valid_pos_l["start_left_x_um"]) ** 2
            + (valid_pos_l["end_left_y_um"] - valid_pos_l["start_left_y_um"]) ** 2
        ) ** 0.5

        distances_r = (
            (valid_pos_r["end_right_x_um"] - valid_pos_r["start_right_x_um"]) ** 2
                + (valid_pos_r["end_right_y_um"] - valid_pos_r["start_right_y_um"]) ** 2
        ) ** 0.5
        
        results[algorithm] = {
            "average_time_s": group["time_taken_s"].mean(),

            "average_measured_uW":
                group["ending_power_mW"].mean()*1000,

            "std_measured_uW":
                group["ending_power_mW"].std(ddof=0)*1000,

            "percentage_off": (1-(group["ending_power_mW"].mean())/(10 ** (measured_max / 10)))*100,

            "ground_truth_max_power": (10 ** (measured_max / 10))*1000,

            "std_x_um": (std_lx + std_rx) / 2,
            "std_y_um": (std_ly + std_ry) / 2,

            "average_distance_travelled_um": (distances_l.mean()+distances_r.mean()) / 2,
        }

    return results


if __name__ == "__main__":
    #results = csv_to_arrays("261020_ID1_peak_search_results", -13.2964668)
    results = csv_to_arrays("261029_ID1_sorted_double_coordinate_peak_search_results", -13.5234642)
    #results = csv_to_arrays("261027_ID3_peak_search_results", -13.6575813)
    #results = csv_to_arrays("261026_ID52_peak_search_results", -60)

    

    for algorithm, stats in results.items():
        print(f"\n--- {algorithm} ---")

        for name, value in stats.items():
            if pd.isna(value):
                print(f"{name}: N/A")
            else:
                print(f"{name}: {value:.4f}")
