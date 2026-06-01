import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from astropy.io import fits
from netCDF4 import Dataset

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
    with Dataset(nc_path, "r") as ds:
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
# Module 1: 轨道时间驱动与过滤核心
# ============================================================

def parse_orbit_row(row: pd.Series, window_start: datetime, window_end: datetime) -> Optional[OrbitLineData]:
    """
    流驱动控制入口。逐行过滤未来时间戳，实现按成像窗口硬过滤，避免无效计算。
    """
    try:
        timestamp = pd.to_datetime(row["Time (YYYY-MM-DD)"])
        
        # 核心逻辑：不在工作窗口内，直接返回 None 触发外层 Skip 机制
        if not (window_start <= timestamp <= window_end):
            return None

        pos = np.array([row["x (km)"], row["y (km)"], row["z (km)"]], dtype=np.float64)
        vel = np.array([row["Derivative x (km/sec)"], row["Derivative y (km/sec)"], row["Derivative z (km/sec)"]], dtype=np.float64)

        return OrbitLineData(timestamp=timestamp, pos_vector_km=pos, vel_vector_kms=vel)
    except Exception as e:
        print(f"[Orbit Parse Warning] 轨道行解析异常: {e}")
        return None

# ============================================================
# Module 2: 空间坐标变换与解算核心 (视线矢量生成)
# ============================================================

def resolve_spatial_geometry(orbit: OrbitLineData, base: StaticBaseData) -> SpatialGeometryResult:
    """
    空间运动学解算层。将物理空间的 3D 向量关系转化为无量纲缩放标量与 3D 视线矢量。
    """
    # 1. 计算卫星至日心的绝对欧氏物理距离
    distance_km = np.linalg.norm(orbit.pos_vector_km)
    if distance_km == 0:
        raise ValueError("零值距离错误：卫星坐标不能与日心原点重合。")

    # 2. 计算无量纲缩放因子 (比例对等)
    scale_factor = base.d_ref_km / distance_km

    # 3. 求解当前帧动态演化后的太阳视半径 (像素)
    solar_radius_pixel = base.rsun_ref_pixel * scale_factor

    # 4. 优雅计算标准化视线单位矢量 [ux, uy, uz]，供未来物理纹理模块调用
    u_view = orbit.pos_vector_km / distance_km

    return SpatialGeometryResult(
        timestamp=orbit.timestamp,
        distance_km=distance_km,
        scale_factor=scale_factor,
        solar_radius_pixel=solar_radius_pixel,
        u_view=u_view
    )

# ============================================================
# Module 3: 二维几何缩放形变引擎
# ============================================================

def scale_single_channel(image: np.ndarray, scale: float) -> np.ndarray:
    """
    独立单通道高保真重采样。通过切片操作严格保证输出画布分辨率的对称性与一致性。
    """
    h, w = image.shape
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # 使用三次插值保证日冕等精细结构重采样时不发生严重失真
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 核心分支自适应分流处理
    if scale >= 1.0:
        # 分支 A：卫星靠近 -> 太阳变大 -> 严格以图像中心执行矩阵裁剪 (Crop)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        result = scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        # 分支 B：卫星远离 -> 太阳变小 -> 四周空缺区域画布填充背景零值 (Padding)
        result = np.zeros((h, w), dtype=image.dtype)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        result[start_y:start_y+new_h, start_x:start_x+new_w] = scaled

    return result

def geometric_scale_engine(base_matrix: np.ndarray, geometry: SpatialGeometryResult) -> ProcessedMatrixData:
    """
    多通道并行几何形变核心引擎。
    """
    scale = geometry.scale_factor
    h, w, c = base_matrix.shape
    scaled_channels = []

    # 维持各物理波段通道（如极紫外 171/193 等）高度内聚、独立运算
    for ch in range(c):
        scaled = scale_single_channel(base_matrix[:, :, ch], scale)
        scaled_channels.append(scaled)

    # 重新在深度方向堆叠回多维科学张量 (H, W, C)
    final_matrix = np.stack(scaled_channels, axis=-1)
    return ProcessedMatrixData(matrix=final_matrix)

# ============================================================
# 预留物理层: 物理纹理与投影校正扩展接口
# ============================================================

def apply_physical_texture_correction(processed: ProcessedMatrixData, geometry: SpatialGeometryResult) -> ProcessedMatrixData:
    """
    [预留物理扩展接口]
    后续做物理纹理工作时，直接在此函数内面向 geometry.u_view 矢量与 geometry.distance_km 进行编程。
    可在不改动前后几何管线的前提下，无缝嵌入边缘昏暗效应（CLD）、多普勒频移或三维球面投影失真修正。
    """
    # 默认执行浅拷贝透传，保障基础几何流的闭环
    return ProcessedMatrixData(matrix=processed.matrix.copy())

# ============================================================
# Module 4: 数据融合与标准化 NetCDF 导出器
# ============================================================

def export_netcdf(processed: ProcessedMatrixData, geometry: SpatialGeometryResult, output_path: str):
    """
    高耦合出口。将重采样矩阵、未来仿真时间戳以及关键的三维视线方向矢量一并打包落盘。
    """
    matrix = processed.matrix
    h, w, c = matrix.shape

    with Dataset(output_path, "w", format="NETCDF4") as ds:
        # 1. 创建维度空间
        ds.createDimension("height", h)
        ds.createDimension("width", w)
        ds.createDimension("channel", c)

        # 2. 写入主图像变量 (32位浮点数科学格式)
        image_var = ds.createVariable("image", "f4", ("height", "width", "channel"))
        image_var[:] = matrix

        # 3. 强行重写关键物理元数据属性，完成未来身份标识注入
        ds.setncattr("simulation_time", geometry.timestamp.isoformat())
        ds.setncattr("distance_km", float(geometry.distance_km))
        ds.setncattr("scale_factor", float(geometry.scale_factor))
        ds.setncattr("solar_radius_pixel", float(geometry.solar_radius_pixel))
        
        # 关键：将当前行算出的视线矢量持久化保存，确保下游消费端的数据可追溯性
        ds.setncattr("view_vector_j2000", geometry.u_view.tolist())

    print(f"[Module 4] 成功导出高保真科学文件: {output_path}")

# ============================================================
# 辅助工具：开箱即用自动化测试数据生成器
# ============================================================

def _generate_mock_assets(fits_p: str, nc_p: str, csv_p: str):
    """检测到文件缺失时自动运行，生成基础高斯太阳模拟数据，保障脚本一键跑通"""
    print("\n[环境检查] 未检测到输入资产文件，正在自动构建轻量化仿真资产...")
    
    # 1. Mock 一个带有关键 Header 的 FITS 基准文件
    hdu = fits.PrimaryHDU()
    hdu.header["DSUN_OBS"] = 149597870700.0  # 1 AU (米)
    hdu.header["RSUN_OBS"] = 959.63
    hdu.header["CDELT1"] = 0.6               # 0.6 arcsec/pixel
    hdu.header["CRPIX1"] = 2048.0
    hdu.header["CRPIX2"] = 2048.0
    hdu.writeto(fits_p, overwrite=True)
    
    # 2. Mock 一个 NetCDF 多通道矩阵 (包含中心高斯斑仿真太阳)
    with Dataset(nc_p, "w", format="NETCDF4") as ds:
        ds.createDimension("height", 4096)
        ds.createDimension("width", 4096)
        ds.createDimension("channel", 3)
        img_var = ds.createVariable("image", "f4", ("height", "width", "channel"))
        
        # 生成一个基础圆盘矩阵作为仿真测试纹理
        y, x = np.ogrid[-2048:2048, -2048:2048]
        mask = (x**2 + y**2) <= 1599**2
        mock_sun = np.zeros((4096, 4096, 3), dtype=np.float32)
        for c in range(3):
            mock_sun[:, :, c] = mask.astype(np.float32) * (1.0 - 0.3 * c)
        img_var[:] = mock_sun

    # 3. Mock 轨道数据 CSV 文件 (包含窗口内及窗口外的时间序列)
    orbit_data = {
        "Time (YYYY-MM-DD)": ["2025-05-12 12:00:00", "2032-06-15 00:00:00", "2040-08-20 18:00:00"],
        "x (km)": [1.49e8, 1.20e8, 1.65e8],
        "y (km)": [0.0, 5.0e7, -3.0e7],
        "z (km)": [0.0, 2.0e6, 1.0e6],
        "Derivative x (km/sec)": [30.0, 25.0, 32.0],
        "Derivative y (km/sec)": [0.0, 2.1, -1.5],
        "Derivative z (km/sec)": [0.0, 0.1, -0.05]
    }
    pd.DataFrame(orbit_data).to_csv(csv_p, index=False)
    print("[环境检查] 仿真测试数据集生成完毕。\n")

# ============================================================
# 主流水线编排调度引擎
# ============================================================

def main():
    # 资产路径配置
    fits_path = "reference.fits"
    nc_path = "template.nc"
    orbit_csv = "trajectory.csv"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # 自动化环境守卫
    if not (os.path.exists(fits_path) and os.path.exists(nc_path) and os.path.exists(orbit_csv)):
        _generate_mock_assets(fits_path, nc_path, orbit_csv)

    # 设定科学研究关注的未来成像窗口
    window_start = datetime(2029, 1, 1)
    window_end = datetime(2044, 12, 31)

    # [Module 0] 基准提取与图像常驻内存
    base_data = init_memory_cache(fits_path, nc_path)

    # 读取高维未来轨道矩阵数据
    orbit_df = pd.read_csv(orbit_csv)

    print(f"\n[主循环开始] 开始遍历主轴未来轨道数据，总计行数: {len(orbit_df)}")
    
    for idx, row in orbit_df.iterrows():
        # [Module 1] 时间维度硬过滤与提取
        orbit_data = parse_orbit_row(row, window_start, window_end)
        if orbit_data is None:
            print(f" -> [Row {idx}] 时间戳不在自定义成像窗口 [{window_start.year}-{window_end.year}] 内，优雅跳过。")
            continue

        # [Module 2] 3D 空间几何与归一化视线向量解析
        geometry = resolve_spatial_geometry(orbit_data, base_data)

        print(f"\n------------------------------------------------------------")
        print(f"[Frame {idx}] 触发处理流 | 目标仿真时刻: {geometry.timestamp}")
        print(f"解算指标 -> 日心物理距离: {geometry.distance_km:.4e} km")
        print(f"解算指标 -> 几何缩放比例 Scale: {geometry.scale_factor:.4f} ({'中心裁剪' if geometry.scale_factor >= 1.0 else '边缘补零'})")
        print(f"核心视线单位矢量 u_view: {geometry.u_view.round(4).tolist()}")

        # [Module 3] 通道解耦二维几何形变变换
        scaled_data = geometric_scale_engine(base_data.matrix_template, geometry)

        # [Future Physics Layer] 挂载预留的未来物理修正层
        final_result = apply_physical_texture_correction(scaled_data, geometry)

        # [Module 4] 混合元数据包装与标准化科学格式落盘
        output_path = os.path.join(
            output_dir, 
            f"solar_sim_{geometry.timestamp.strftime('%Y%m%d_%H%M%S')}.nc"
        )
        export_netcdf(final_result, geometry, output_path)

    print("\n[工程完工] 流水线执行完毕，全部生成帧已安全落盘。")

if __name__ == "__main__":
    main()