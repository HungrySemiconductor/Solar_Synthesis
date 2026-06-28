import os
from datetime import datetime

import pandas as pd

import filter_txt_to_csv
import single_scale


# =========================
# 路径配置
# =========================

# __file__ 是 Python 自动提供的变量，表示“当前这个 .py 文件的路径”。
# os.path.abspath(__file__) 会把它变成绝对路径。
# os.path.dirname(...) 会取出这个文件所在的文件夹。
#
# 例如当前文件是：
# D:\Solar_Images\Solar_Scaler\src\batch_scale.py
#
# 那么 BASE_DIR 就是：
# D:\Solar_Images\Solar_Scaler\src
#
# 这样做的好处是：无论你从哪个目录运行这个脚本，
# 下面的输入/输出文件都能按照相对 src 的位置被找到。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始 SPO 轨道参数 TXT 文件。
# os.path.join(...) 用来拼接路径，比手写斜杠更安全。
# "../data" 表示从 src 文件夹返回上一级，再进入 data 文件夹。
INPUT_TXT = os.path.join(BASE_DIR, "../data", "Inputs", "TXT", "spo_orbit_param_all.txt")

# filter_txt_to_csv.py 清洗后生成的 CSV 文件。
# 如果这个文件已经存在，后面默认会直接读取它，避免重复清洗 TXT。
OUTPUT_CSV = os.path.join(BASE_DIR, "../data", "Outputs", "CSV", "batch_scaled.csv")

# 要被重复使用的原始 NetCDF 文件。
# 批处理时，这个 nc 文件只加载一次，然后循环生成不同时间/距离下的结果。
INPUT_NC = os.path.join(BASE_DIR, "../data", "Inputs", "NetCDF", "20141023_1000.nc")

# 批量缩放结果的输出文件夹。
OUTPUT_DIR = os.path.join(BASE_DIR, "../data", "Outputs", "NetCDF", "batch_scaled")

# 调试用参数：
# - None 表示处理 CSV 中的所有行
# - 3 表示只处理前 3 行
#
# 因为每处理一行都会生成一个新的 nc 文件，而且文件可能很大，
# 第一次测试时建议先改成 MAX_ROWS = 3。
MAX_ROWS = None


# =========================
# 成像时间窗口
# =========================

# 这里定义的是 filter_txt_to_csv.py 用来筛选轨道数据的时间范围。
#
# 列表 IMAGING_WINDOWS 中的每一项都是一个元组 tuple：
#     (开始时间, 结束时间)
#
# datetime(2030, 1, 31, 16, 23) 表示：
#     2030 年 1 月 31 日 16 点 23 分 00 秒
#
# filter_txt_to_csv.py 会检查 TXT 中每一行的 timestamp，
# 只有落在这些时间窗口内的数据才会被写入 CSV。
#
# 取代表性时间窗口，各取1个时间点
# 因为时间精度为（xx时 23分 17.04秒），所以下面每个时间段只包含1个时间点
IMAGING_WINDOWS = [
    (datetime(2030, 1, 31, 16, 23), datetime(2030, 1, 31, 17, 23)),
    (datetime(2031, 1, 31, 16, 23), datetime(2031, 1, 31, 17, 23)),

    (datetime(2035, 1, 1, 16, 23), datetime(2035, 1, 1, 17, 23)),
    (datetime(2036, 7, 15, 16, 23), datetime(2036, 7, 15, 17, 23)),

    (datetime(2038, 1, 12, 16, 23), datetime(2038, 1, 12, 17, 23)),
    (datetime(2039, 1, 10, 16, 23), datetime(2039, 1, 10, 17, 23)),

    (datetime(2040, 1, 1, 16, 23), datetime(2040, 1, 1, 17, 23)),
    (datetime(2042, 4, 14, 16, 23), datetime(2042, 4, 14, 17, 23)),
]


def ensure_orbit_csv(txt_path, csv_path, imaging_windows, overwrite=False):
    """
    确保轨道 CSV 文件存在，并把 CSV 读取成 pandas DataFrame。

    你可以把 DataFrame 理解成 Python 里的“表格”：
    - 每一行是一条轨道记录
    - 每一列是一种参数，比如 timestamp、x_km、distance_km

    参数：
    txt_path:
        原始轨道 TXT 文件路径。

    csv_path:
        清洗后的 CSV 文件路径。

    imaging_windows:
        成像时间窗口，只保留这些窗口内的数据。

    overwrite:
        是否强制重新生成 CSV。
        - False：如果 CSV 已存在，就直接读取旧 CSV
        - True：即使 CSV 已存在，也重新从 TXT 清洗生成

    返回值：
        pandas.DataFrame，也就是清洗后的轨道参数表。
    """

    # os.path.exists(csv_path) 用来判断文件是否存在。
    #
    # 这个 if 的逻辑是：
    # 如果 overwrite=True，或者 CSV 文件不存在，
    # 就调用 filter_txt_to_csv.py 从 TXT 重新生成 CSV。
    if overwrite or not os.path.exists(csv_path):
        return filter_txt_to_csv.filter_txt_to_csv(
            txt_path=txt_path,
            output_csv_path=csv_path,
            imaging_windows=imaging_windows,
        )

    # 如果 CSV 已经存在，并且 overwrite=False，
    # 就直接读取已有 CSV，节省时间。
    print("\n===================================")
    print("Reading Existing Orbit CSV...")
    print("Input CSV:", csv_path)

    # pd.read_csv(...) 会把 CSV 文件读取成 DataFrame。
    return pd.read_csv(csv_path)


def format_time_for_nc(timestamp):
    """
    把 CSV 中的时间转换成 NetCDF 属性需要的时间字符串。

    输入可能长这样：
        2030-01-31 16:23:00

    输出会变成：
        2030-01-31T16:23:00.000

    中间的 T 是常见的 ISO 时间格式写法。
    """

    # pd.to_datetime(...) 可以把字符串、pandas 时间对象等统一转成时间类型。
    ts = pd.to_datetime(timestamp)

    # strftime(...) 用来把时间格式化成指定样式的字符串。
    #
    # 常用格式符：
    # %Y 年，四位数
    # %m 月，两位数
    # %d 日，两位数
    # %H 小时，24 小时制
    # %M 分钟
    # %S 秒
    # %f 微秒，六位数
    #
    # [:-3] 是字符串切片，表示“去掉最后 3 个字符”。
    # 因为 %f 会生成 6 位微秒，例如 .123456；
    # 去掉最后 3 位后变成 .123，也就是毫秒精度。
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def build_output_path(output_dir, timestamp, distance_km):
    """
    根据时间和距离生成输出 nc 文件路径。

    例如：
    时间 timestamp:
        2030-01-31 16:23:00

    距离 distance_km:
        344075101

    文件名会类似：
        scaled_output_20300131_162300_000_344075101km_spo.nc
    """

    ts = pd.to_datetime(timestamp)

    # 把时间放到文件名中。
    # 注意：Windows 文件名中不能有冒号 :，
    # 所以这里使用 20300131_162300_000 这种格式。
    time_part = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # f"{distance_km:.0f}" 是 f-string 格式化写法。
    # :.0f 表示保留 0 位小数。
    distance_part = f"{distance_km:.0f}km"

    # f-string 中的 {变量名} 会被替换成变量当前的值。
    filename = f"scaled_output_{time_part}_{distance_part}_spo.nc"

    return os.path.join(output_dir, filename)


def batch_scale_from_csv(csv_df, input_nc, output_dir, max_rows=None):
    """
    批量缩放主函数。

    这个函数完成的流程：
    1. 检查 CSV 表格中是否有 timestamp 和 distance_km 两列
    2. 清理掉时间或距离为空的行
    3. 打开同一个原始 nc 文件
    4. 从 nc 文件中提取 171、193、304 等目标通道
    5. 遍历 CSV 的每一行
    6. 每一行都用自己的时间和距离生成一个新的 nc 文件

    参数：
    csv_df:
        轨道参数 DataFrame。

    input_nc:
        原始 NetCDF 文件路径。

    output_dir:
        输出文件夹路径。

    max_rows:
        最多处理多少行。
        None 表示不限制。
    """

    # set 是 Python 的“集合”类型。
    # 集合适合用来判断“某些元素是否存在”。
    #
    # 这里表示 CSV 必须包含这两列：
    # - timestamp：时间
    # - distance_km：SPO 到日心距离，单位 km
    required_columns = {"timestamp", "distance_km"}

    # csv_df.columns 是当前表格实际拥有的所有列名。
    #
    # required_columns - set(csv_df.columns)
    # 表示“必须有，但是实际没有”的列。
    missing_columns = required_columns - set(csv_df.columns)

    # 如果缺少必要列，程序无法继续，所以主动抛出错误。
    if missing_columns:
        raise RuntimeError(f"Orbit CSV missing columns: {sorted(missing_columns)}")

    # dropna(...) 会删除指定列中有空值的行。
    # subset=["timestamp", "distance_km"] 表示只检查这两列。
    #
    # .copy() 表示复制一份新的 DataFrame，
    # 避免后续操作影响原来的表格。
    csv_df = csv_df.dropna(subset=["timestamp", "distance_km"]).copy()

    # 如果 max_rows 不是 None，就只处理前 max_rows 行。
    # head(3) 的意思是取前 3 行。
    if max_rows is not None:
        csv_df = csv_df.head(max_rows)

    print("\n===================================")
    print("Batch Scaling Started")
    print("Input NC:", input_nc)
    print("Output Dir:", output_dir)
    print("Rows To Process:", len(csv_df))

    # 打开原始 NetCDF 文件。
    #
    # 注意：这个操作放在循环外面。
    # 因为所有输出都使用同一个原始 nc 文件，
    # 如果每一行都重新打开一次，会很慢。
    ds = single_scale.load_nc_with_xarray(input_nc)

    # try/finally 是一种“保证收尾”的写法。
    #
    # try 里的代码正常运行或中途报错后，
    # finally 里的代码都会执行。
    #
    # 这里 finally 里会关闭 ds，避免 nc 文件一直被占用。
    try:
        # 从原始 nc 中提取需要处理的通道。
        #
        # single_scale.TARGET_WAVELENGTHS 一般类似 [171, 193, 304]。
        #
        # channels 是图像数组列表。
        # names 是波段名称列表。
        channels, names = single_scale.extract_target_channels(
            ds,
            single_scale.TARGET_WAVELENGTHS,
        )

        # iterrows() 用来逐行遍历 DataFrame。
        #
        # row_index 是这一行的索引。
        # row 是这一行的数据，可以像字典一样通过列名取值：
        #     row["timestamp"]
        #     row["distance_km"]
        for row_index, row in csv_df.iterrows():
            # 当前行的时间，转换成 nc 文件属性中使用的字符串格式。
            custom_time = format_time_for_nc(row["timestamp"])

            # 当前行的距离。
            # float(...) 把它转换成浮点数，避免从 CSV 读出来时是字符串。
            #
            # 这个值会作为 single_scale.py 里的 custom_dsun_obs 使用。
            custom_dsun_obs = float(row["distance_km"])

            # 当前行对应的输出 nc 文件路径。
            output_nc = build_output_path(output_dir, row["timestamp"], custom_dsun_obs)

            print("\n===================================")
            print(f"Processing Row: {row_index}")
            print("Observation Time:", custom_time)
            print("Custom dsun_obs:", custom_dsun_obs, "km")

            # 根据原始 nc 文件中的 dsun_obs 和当前行的 distance_km
            # 计算缩放因子。
            #
            # 简单理解：
            # - 距离越远，太阳在图像中看起来越小
            # - 距离越近，太阳在图像中看起来越大
            scale_factor = single_scale.calculate_scale_factor(ds, custom_dsun_obs)

            # 对每个目标通道做图像缩放。
            scaled_channels = single_scale.scale_channels(channels, scale_factor)

            # 把缩放后的图像重新组织成一个新的 xarray Dataset。
            #
            # custom_time 会被写进输出 nc 的属性中，
            # 表示这个模拟图像对应的观测时间。
            ds_out = single_scale.build_output_dataset(
                scaled_channels=scaled_channels,
                wavelengths=names,
                scale_factor=scale_factor,
                custom_time=custom_time,
            )

            # 给输出 nc 增加一些轨道来源信息。
            # 这些信息不是缩放本身必须的，但方便以后追踪这个文件怎么来的。
            ds_out.attrs["source_orbit_distance_km"] = custom_dsun_obs

            # 这个 for 循环会依次检查下面这些列是否存在：
            # x_km、y_km、z_km、vx_kms、vy_kms、vz_kms
            #
            # 如果存在，就写入输出 nc 的全局属性。
            for column in ["x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"]:
                if column in row:
                    ds_out.attrs[f"source_orbit_{column}"] = float(row[column])

            # 保存当前这一行对应的输出 nc 文件。
            single_scale.save_dataset(ds_out, output_nc)

            # 关闭当前输出 Dataset，释放资源。
            ds_out.close()

    finally:
        # 无论前面是否成功，都关闭输入 nc。
        ds.close()

    print("\n===================================\nBATCH ALL DONE")


# =========================
# 脚本入口
# =========================

# 这是一种 Python 常见写法。
#
# 当你直接运行：
#     python batch_scale.py
#
# 下面的代码会执行。
#
# 但如果别的文件只是：
#     import batch_scale
#
# 下面的代码不会自动执行。
if __name__ == "__main__":
    # 第一步：
    # 确保轨道 CSV 存在，并读取成 DataFrame。
    orbit_df = ensure_orbit_csv(
        txt_path=INPUT_TXT,
        csv_path=OUTPUT_CSV,
        imaging_windows=IMAGING_WINDOWS,
        overwrite=False,
    )

    # 第二步：
    # 用 CSV 中的每一行驱动 single_scale.py 的缩放流程。
    #
    # 也就是：
    # - 同一个 input_nc
    # - 不同 timestamp
    # - 不同 distance_km
    # - 输出多个不同的 nc 文件
    batch_scale_from_csv(
        csv_df=orbit_df,
        input_nc=INPUT_NC,
        output_dir=OUTPUT_DIR,
        max_rows=MAX_ROWS,
    )
