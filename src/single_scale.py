import os
import numpy as np
import xarray as xr
import json

from scipy.ndimage import zoom
from datetime import datetime

INPUT_NC = "data/Inputs/NetCDF/20141023_1000.nc"
OUTPUT_NC = "data/Outputs/NetCDF/scaled_output_0.83_spo.nc"
# OUTPUT_NC = "data/Outputs/NetCDF/scaled_output_3.2_spo.nc"

# 固定缩放因子
# SCALE_FACTOR = 0.83

# 自定义观测距离（单位：千米）
CUSTOM_DSUN_OBS = 124166232 # 0.83AU
# CUSTOM_DSUN_OBS = 478713184 # 3.2AU

# # 太阳视半径（arcsec）
# CUSTOM_RSUN_OBS = 1200.0

# 目标通道
TARGET_WAVELENGTHS = [171, 193, 304]

# 自定义时间变量
CUSTOM_TIME_STR = "2026-05-27T12:00:00.000" 


# 加载 NetCDF 文件到 xarray Dataset
def load_nc_with_xarray(nc_path):
    print("\n===================================")
    print("Loading NetCDF with xarray...")
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    return ds

# 计算缩放因子
def calculate_scale_factor(ds, custom_dsun_obs):
    print("\n===================================")
    print("Calculating Scale Factor...")
    
    # 从通道变量的meta属性中提取dsun_obs
    dsun_obs_m = None
    
    for wave in TARGET_WAVELENGTHS:
        var_name = f"aia{wave}"
        if var_name not in ds:
            continue
            
        var = ds[var_name]
        
        # 尝试从meta_0或meta_1中提取
        for meta_key in ['meta_0', 'meta_1']:
            if meta_key in var.attrs:
                try:
                    meta_str = var.attrs[meta_key]
                    header_dict = json.loads(meta_str)
                    if "dsun_obs" in header_dict:
                        dsun_obs_m = header_dict["dsun_obs"]
                        print(f"Found dsun_obs in {var_name}.{meta_key}: {dsun_obs_m}")
                        break
                except (json.JSONDecodeError, TypeError):
                    continue
        
        if dsun_obs_m is not None:
            break
    
    if dsun_obs_m is None:
        raise RuntimeError("Cannot find 'dsun_obs' in dataset variables or attributes")
    
    dsun_obs_km = dsun_obs_m / 1000  # 转为km

    scale_factor = dsun_obs_km / custom_dsun_obs
    
    print(f"dsun_obs from data: {dsun_obs_m}m = {dsun_obs_km}km")
    print(f"Custom dsun_obs: {custom_dsun_obs}km")
    print(f"Calculated scale_factor: {scale_factor}")
    
    return scale_factor

# 提取目标通道
def extract_target_channels(ds, target_wavelengths):
    print("\n===================================")
    print("Extracting Channels...")
    extracted_channels = []
    extracted_names = []

    for wave in target_wavelengths:
        var_name = f"aia{wave}"
        if var_name not in ds:
            raise RuntimeError(f"Cannot find variable: {var_name}")
        
        # 保持 xarray 读取出来的浮点状态
        channel = ds[var_name].values.astype(np.float32)
        extracted_channels.append(channel)
        extracted_names.append(wave)
        print(f"Extracted: {var_name} {channel.shape}")

    return extracted_channels, extracted_names

# 缩放单通道
def scale_single_channel(
    image,
    scale_factor
):

    h, w = image.shape

    # ==========================================
    # 提取真实背景值，用随机数填充
    # ==========================================
    # edge_pixels = np.concatenate([
    #     image[:10,:].flatten(),
    #     image[-10:,:].flatten(),
    #     image[:, :10].flatten(),
    #     image[:, -10:].flatten()
    #  ])
    # background_value = np.median(edge_pixels)

    # 用随机数填充背景值
    background_value = np.random.uniform(0.0, 0.1, size=(h,w))

   
    # scipy做缩放处理
    scaled = zoom(image, zoom=scale_factor, order=3)

    sh, sw = scaled.shape

    # 放大处理
    if scale_factor >= 1.0:
        start_y = (sh - h) // 2
        start_x = (sw - w) // 2

        result = scaled[
            start_y:start_y+h,
            start_x:start_x+w
        ]

    # 缩小处理
    else:
        result = np.full(
            (h,w),
            background_value,
            dtype=np.float32
        )

        start_y = (h - sh) // 2
        start_x = (w - sw) // 2

        result[
            start_y:start_y+sh,
            start_x:start_x+sw
        ] = scaled

    return result

# 循环缩放处理所有通道
def scale_channels(channels, scale_factor):
    print("\n===================================")
    print("Scaling Channels...")
    scaled_channels = []
    for i, ch in enumerate(channels):
        print(f"Scaling Channel {i}")
        scaled = scale_single_channel(ch, scale_factor)
        scaled_channels.append(scaled)
    return scaled_channels


# 构建输出 Dataset
def build_output_dataset(scaled_channels, wavelengths, scale_factor, custom_time):
    print("\n===================================")
    print("Building Output Dataset with Custom Time...")
    data_vars = {}

    for channel_data, wave in zip(scaled_channels, wavelengths):
        var_name = f"aia{wave}"
        
        # 将缩放后的矩阵包装为带“自定义时间属性”的 DataArray
        data_vars[var_name] = (
            ["y", "x"], 
            channel_data,
            {
                "unit": "DN/s",
                "t_obs": custom_time,  # 未来观测时间
                "scale_factor": scale_factor,
                "description": f"Simulated Level-1.5 SPO image for wavelength {wave} Angstrom",
                "_ChunkSizes": "4096U, 4096U"
            }
        )

    # 创建全局 Dataset
    ds_out = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "title": "Scaled Solar Disk Dataset;",
            "scale_factor": scale_factor,
            "observation_time": custom_time,  # 未来观测时间
            "production_time": datetime.now().isoformat()  # 生产时间
        }
    )
    return ds_out


def save_dataset(ds_out, output_path):
    print("\n===================================")
    print("Saving Dataset...")
    output_dir = os.path.dirname(output_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    # 强制指定底层 HDF5/NetCDF 的 _FillValue 属性 ---
    # 显式告诉 Panoply：遇到 NaN 请将其作为无效值图层处理
    encoding = {}
    for var_name in ds_out.data_vars:
        encoding[var_name] = {
            "_FillValue": np.nan,  
            "dtype": "float32"
        }

    ds_out.to_netcdf(
        output_path,
        engine="h5netcdf",
        encoding=encoding # 传入编码配置
    )
    print("Saved:", output_path)


if __name__ == "__main__":

    ds = load_nc_with_xarray(INPUT_NC)

    scale_factor = calculate_scale_factor(ds, CUSTOM_DSUN_OBS)
    channels, names = extract_target_channels(ds, TARGET_WAVELENGTHS)
    scaled_channels = scale_channels(channels, scale_factor)
    
    ds_out = build_output_dataset(scaled_channels, names, scale_factor, CUSTOM_TIME_STR)
    
    save_dataset(ds_out, OUTPUT_NC)
    print("\n===================================\nALL DONE")