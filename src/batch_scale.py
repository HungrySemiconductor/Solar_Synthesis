import os
from datetime import datetime

import pandas as pd

import filter_txt_to_csv
import single_scale


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TXT = os.path.join(BASE_DIR, "data", "spo_orbit_param_all.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "spo_orbit_param_valid.csv")
INPUT_NC = os.path.join(BASE_DIR, "data", "NetCDF", "20141023_1000.nc")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Outputs", "batch_scaled")

# Set to an integer when testing, for example 3. Use None to process all rows.
MAX_ROWS = None

IMAGING_WINDOWS = [
    (datetime(2029, 12, 19), datetime(2029, 12, 20)),
    (datetime(2031, 1, 31), datetime(2031, 2, 1)),
    (datetime(2035, 1, 1), datetime(2035, 1, 2)),
    (datetime(2036, 7, 15), datetime(2036, 7, 16)),
    (datetime(2038, 1, 12), datetime(2038, 1, 13)),
    (datetime(2039, 1, 10), datetime(2039, 1, 11)),
    (datetime(2040, 1, 1), datetime(2040, 1, 2)),
    (datetime(2042, 4, 14), datetime(2042, 4, 15)),
]


def ensure_orbit_csv(txt_path, csv_path, imaging_windows, overwrite=False):
    """Create the orbit CSV when needed, then return it as a DataFrame."""
    if overwrite or not os.path.exists(csv_path):
        return filter_txt_to_csv.filter_txt_to_csv(
            txt_path=txt_path,
            output_csv_path=csv_path,
            imaging_windows=imaging_windows,
        )

    print("\n===================================")
    print("Reading Existing Orbit CSV...")
    print("Input CSV:", csv_path)
    return pd.read_csv(csv_path)


def format_time_for_nc(timestamp):
    ts = pd.to_datetime(timestamp)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def build_output_path(output_dir, timestamp, distance_km):
    ts = pd.to_datetime(timestamp)
    time_part = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    distance_part = f"{distance_km:.0f}km"
    filename = f"scaled_output_{time_part}_{distance_part}_spo.nc"
    return os.path.join(output_dir, filename)


def batch_scale_from_csv(csv_df, input_nc, output_dir, max_rows=None):
    required_columns = {"timestamp", "distance_km"}
    missing_columns = required_columns - set(csv_df.columns)
    if missing_columns:
        raise RuntimeError(f"Orbit CSV missing columns: {sorted(missing_columns)}")

    csv_df = csv_df.dropna(subset=["timestamp", "distance_km"]).copy()
    if max_rows is not None:
        csv_df = csv_df.head(max_rows)

    print("\n===================================")
    print("Batch Scaling Started")
    print("Input NC:", input_nc)
    print("Output Dir:", output_dir)
    print("Rows To Process:", len(csv_df))

    ds = single_scale.load_nc_with_xarray(input_nc)
    try:
        channels, names = single_scale.extract_target_channels(ds, single_scale.TARGET_WAVELENGTHS)

        for row_index, row in csv_df.iterrows():
            custom_time = format_time_for_nc(row["timestamp"])
            custom_dsun_obs = float(row["distance_km"])
            output_nc = build_output_path(output_dir, row["timestamp"], custom_dsun_obs)

            print("\n===================================")
            print(f"Processing Row: {row_index}")
            print("Observation Time:", custom_time)
            print("Custom dsun_obs:", custom_dsun_obs, "km")

            scale_factor = single_scale.calculate_scale_factor(ds, custom_dsun_obs)
            scaled_channels = single_scale.scale_channels(channels, scale_factor)
            ds_out = single_scale.build_output_dataset(
                scaled_channels=scaled_channels,
                wavelengths=names,
                scale_factor=scale_factor,
                custom_time=custom_time,
            )

            ds_out.attrs["source_orbit_distance_km"] = custom_dsun_obs
            for column in ["x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"]:
                if column in row:
                    ds_out.attrs[f"source_orbit_{column}"] = float(row[column])

            single_scale.save_dataset(ds_out, output_nc)
            ds_out.close()

    finally:
        ds.close()

    print("\n===================================\nBATCH ALL DONE")


if __name__ == "__main__":
    orbit_df = ensure_orbit_csv(
        txt_path=INPUT_TXT,
        csv_path=OUTPUT_CSV,
        imaging_windows=IMAGING_WINDOWS,
        overwrite=False,
    )

    batch_scale_from_csv(
        csv_df=orbit_df,
        input_nc=INPUT_NC,
        output_dir=OUTPUT_DIR,
        max_rows=MAX_ROWS,
    )
