import os

import h5netcdf

import numpy as np
from datetime import datetime
from dataclasses import dataclass

from astropy.io import fits


# ============================================================
# 强类型数据契约定义 (Data Contracts)
# ============================================================

@dataclass(frozen=True)
class StaticBaseData:
    """Module 0 输出：全程序只读一次、常驻内存的静态几何母版数据"""
    matrix_template: np.ndarray       # 形状为 (H, W, C) 的原始图像矩阵
    d_ref_km: float                   # 转换为公里单位的基准视在距离 (DSUN_OBS / 1000)
    rsun_obs_arcsec: float            # 原始日面视半径角秒基准
    cdelt_arcsec_per_pixel: float     # 原始像素分辨率 (arcsec/pixel)
    rsun_ref_pixel: float             # 基准状态下的太阳像素半径
    solar_center_x: float             # 原始图像中的太阳中心 X 坐标
    solar_center_y: float             # 原始图像中的太阳中心 Y 坐标

@dataclass(frozen=True)
class OrbitLineData:
    """Module 1 输出：经时间窗口过滤后的单条合法未来轨道状态"""
    timestamp: datetime
    pos_vector_km: np.ndarray         # J2000 日心坐标向量 [x, y, z]，单位：km
    vel_vector_kms: np.ndarray        # 速度矢量 [vx, vy, vz]，单位：km/s

@dataclass(frozen=True)
class SpatialGeometryResult:
    """Module 2 输出：三维空间运动学与观测几何解算结果（核心解耦层）"""
    timestamp: datetime
    distance_km: float                # 卫星到日心的绝对物理距离 (km)
    scale_factor: float               # 精确、无量纲的几何缩放因子 (D_ref / D_SPO)
    solar_radius_pixel: float         # 变形后的太阳像素半径
    u_view: np.ndarray                # 核心接口：归一化的 3D 视线单位向量 [ux, uy, uz]
    
    # 未来物理纹理与载荷姿态扩展接口预留
    roll_angle_deg: float = 0.0       # 卫星绕视线自转偏航角
    off_axis_angle_deg: float = 0.0   # 离轴偏角

@dataclass(frozen=True)
class ProcessedMatrixData:
    """Module 3 输出：仅完成二维几何变换与画布自适应处理后的图像矩阵"""
    matrix: np.ndarray                # 严格保持 (4096, 4096, C) 的多通道图像矩阵

# ============================================================
# Module 0: 初始化与内存缓存管理器
# ============================================================

def init_memory_cache(fits_path: str, nc_path: str) -> StaticBaseData:
    """
    全程序仅执行一次。解析静态母版元数据，并将海量图像矩阵常驻内存以消灭磁盘 I/O 瓶颈。
    """
    print(f"\n[Module 0] 正在读取基准 FITS 文件: {fits_path}")
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        d_ref_km = header["DSUN_OBS"] / 1000.0  # 统一米至公里单位
        rsun_obs_arcsec = header["RSUN_OBS"]
        cdelt = header["CDELT1"]
        solar_center_x = header["CRPIX1"]
        solar_center_y = header["CRPIX2"]
        rsun_ref_pixel = rsun_obs_arcsec / cdelt

    print(f"[Module 0] 正在缓存多通道 NetCDF 母版矩阵: {nc_path}")
    with h5netcdf.Dataset(nc_path, "r") as ds:
        # 兼容读取可能存在的 (C, H, W) 或 (H, W, C) 格式
        matrix_template = ds.variables["image"][:]
    
    matrix_template = np.array(matrix_template, dtype=np.float32)
    
    # 自动纠正维度顺序，确保 OpenCV 几何算子面向标准的 (H, W, C)
    if matrix_template.shape[0] < 20:
        matrix_template = np.transpose(matrix_template, (1, 2, 0))

    print(f"[Module 0] 缓存成功 | 矩阵形状: {matrix_template.shape} | 基准距离: {d_ref_km:.2f} km")
    return StaticBaseData(
        matrix_template=matrix_template,
        d_ref_km=d_ref_km,
        rsun_obs_arcsec=rsun_obs_arcsec,
        cdelt_arcsec_per_pixel=cdelt,
        rsun_ref_pixel=rsun_ref_pixel,
        solar_center_x=solar_center_x,
        solar_center_y=solar_center_y
    )

    
# ============================================================
# 主流水线编排调度引擎
# ============================================================

def main():
    # 资产路径配置
    
    FITS_PATH = "FITS/aia.lev1_euv_12s.2014-10-23T100012Z.171.spikes.fits"
    NC_PATH   = "NetCDF/20141023_1000.nc"

    orbit_csv = "spo_trajectory.csv"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

   
if __name__ == "__main__":
    main()