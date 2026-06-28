import os
from datetime import datetime

import pandas as pd

import filter_txt_to_csv
import single_scale


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TXT = os.path.join(BASE_DIR, "../data", "Inputs", "TXT", "spo_orbit_param_all.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "../data", "Outputs", "CSV", "batch_scaled.csv")
INPUT_NC = os.path.join(BASE_DIR, "../data", "Inputs", "NetCDF", "20141023_1000.nc")
OUTPUT_DIR = os.path.join(BASE_DIR, "../data", "Outputs", "NetCDF", "batch_scaled")

# Set to an integer when testing, for example 3. Use None to process all rows.
MAX_ROWS = None

# 日为单位，单日间隔1小时的连续24张图像
# IMAGING_WINDOWS = [
#     (datetime(2029, 12, 19), datetime(2029, 12, 20)),
#     (datetime(2031, 1, 31), datetime(2031, 2, 1)),
#     (datetime(2035, 1, 1), datetime(2035, 1, 2)),
#     (datetime(2036, 7, 15), datetime(2036, 7, 16)),
#     (datetime(2038, 1, 12), datetime(2038, 1, 13)),
#     (datetime(2039, 1, 10), datetime(2039, 1, 11)),
#     (datetime(2040, 1, 1), datetime(2040, 1, 2)),
#     (datetime(2042, 4, 14), datetime(2042, 4, 15)),
# ]

# 小时为单位，单日间隔1小时的连续1张图像，共8张
IMAGING_WINDOWS = [
    (datetime(2030, 1, 31, 16, 23), datetime(2030, 1, 31, 17, 23)),
    (datetime(2031, 1, 31, 16,23), datetime(2031, 1, 31, 17, 23)),

    (datetime(2035, 1, 1, 16,23), datetime(2035, 1, 1, 17, 23)),
    (datetime(2036, 7, 15, 16,23), datetime(2036, 7, 15, 17, 23)),

    (datetime(2038, 1, 12, 16,23), datetime(2038, 1, 12, 17, 23)),
    (datetime(2039, 1, 10, 16,23), datetime(2039, 1, 10, 17, 23)),

    (datetime(2040, 1, 1, 16,23), datetime(2040, 1, 1, 17, 23)),
    (datetime(2042, 4, 14, 16,23), datetime(2042, 4, 14, 17, 23))
    ]

# csv文件不存在时，创建csv文件
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

# 格式化时间，符合nc文件属性的格式，保留到3位毫秒
def format_time_for_nc(timestamp):
    ts = pd.to_datetime(timestamp)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] 

# 构建输出文件路径
def build_output_path(output_dir, timestamp, distance_km):
    ts = pd.to_datetime(timestamp)
    time_part = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    distance_part = f"{distance_km:.0f}km"
    filename = f"scaled_output_{time_part}_{distance_part}_spo.nc"
    return os.path.join(output_dir, filename)

# ====================纹理处理（矢量接口预留）====================
def apply_physical_model(image, geometry_info):
    """
    视角几何接口

    Parameters
    ----------
    image : np.ndarray
        已完成距离缩放后的图像

    geometry_info : dict
        轨道几何信息

        {
            "x_km": ...,
            "y_km": ...,
            "z_km": ...,
            "distance_km": ...,
            "vx": ...,
            "vy": ...,
            "vz": ...
        }

    Returns
    -------
    image : np.ndarray
    """

    """
    物理模型接口

    当前:
        直接返回原图

    未来:
        Diffusion预测纹理变化

    返回:
        物理变化后的太阳图像
    """
    return image

# ====================缩放处理====================
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

            # ====================纹理处理（矢量接口预留）====================
            # geometry_info = {
            # "distance_km": custom_dsun_obs,
            # "x_km": row.get("x_km"),
            # "y_km": row.get("y_km"),
            # "z_km": row.get("z_km"),
            # "vx_kms": row.get("vx_kms"),
            # "vy_kms": row.get("vy_kms"),
            # "vz_kms": row.get("vz_kms")
            # }
            
            # processed_channels = []

            # for ch in channels:
            #     ch = apply_physical_model(
            #         ch,
            #         geometry_info
            #     )
            #     processed_channels.append(ch)

            # 用纹理处理过的processed_channels代替原本nc文件中的channels
            # scaled_channels = single_scale.scale_channels(processed_channels, scale_factor)
            
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
        # 无论前面是否成功，都关闭输入 nc，避免占用资源
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
