import pandas as pd
import numpy as np

from datetime import datetime

def filter_txt_to_csv(txt_path, output_csv_path, imaging_windows):

    print("\n===================================")
    print("Reading SPO Orbit File...")
    print("Input TXT:", txt_path)

    # 找到数据开始行
    start_line = None

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Time (YYYY-MM-DD)" in line:
            # 数据从下一行开始
            start_line = i + 2
            break

    if start_line is None:
        raise RuntimeError(
            "Cannot find orbit table header."
        )
    print("Data Starts At Line:", start_line)

    # 存储清理后的行
    cleaned_rows = []
    
    # 读取每个轨道行
    for line in lines[start_line:]:
        line = line.strip()     # 移除首尾空格
        if len(line) == 0:      # 跳过空行
            continue
        try:
            parts = line.split()     # 按空格分隔开不同参数

            # 将时间参数转换成格式为 YYYY-MM-DD-%H:%M:%S.%f 的 datetime 对象
            time_str = parts[0]
            timestamp = datetime.strptime(time_str, "%Y-%m-%d-%H-%M-%S.%f")

            # 跳过不在成像时间窗口内的行
            in_window = False
            for win_start, win_end in imaging_windows:
                if win_start <= timestamp <= win_end:
                    in_window = True
                    break
            if not in_window:
                continue

            # 转换位置参数为浮点数
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            # 转换速度参数为浮点数
            vx = float(parts[4])
            vy = float(parts[5])
            vz = float(parts[6])

            # 计算SPO到日心的距离
            distance_km = np.sqrt(x**2 + y**2 + z**2)

            # # 从SPO指向日心的单位向量
            # u_view = np.array([x, y, z]) / distance_km

            # 存储清理后的参数
            cleaned_rows.append({
                "timestamp": timestamp,

                "x_km": x,
                "y_km": y,
                "z_km": z,

                "vx_kms": vx,
                "vy_kms": vy,
                "vz_kms": vz,

                "distance_km": distance_km

                # "u_view_x": u_view[0],
                # "u_view_y": u_view[1],
                # "u_view_z": u_view[2]
            })

        except Exception as e:

            print("\n[WARNING]")
            print("Failed Line:")
            print(line)
            print("Reason:", e)

            continue

    # 将清理后的行转换为 DataFrame
    df = pd.DataFrame(cleaned_rows)
    # 保存为 CSV 文件
    df.to_csv(output_csv_path, index=False)

    print("\n===================================")
    print("Orbit Parameters Cleaning Finished")
    print("Output CSV:", output_csv_path)
    print("Total Valid Rows:", len(df))

    return df

# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":
    input_txt = "../data/Inputs/TXT/spo_orbit_param_all.txt"
    output_csv = "../data/Outputs/CSV/spo_orbit_values.csv"

    # 成像窗口 取代表性数据
    '''
    - 地地转移阶段 2029 - 2031 
        2031-01-31 - 2031-02-01 近日点0.83AU  124,166,232
        2029-12-19 - 2029-12-20 远日点2.3AU   344,075,101
    - 极轨探测阶段 
    2035-01-18  
        2035-01-01 - 2035-01-02 近日点1.00AU  149,597,870
        2036-07-15 - 2036-07-16 远日点3.2AU   478,713,184  
    2038-01-18  
        2038-01-12 - 2038-01-13 近日点0.98AU 146,605,912
        2039-01-10 - 2039-01-11 远日点2.2AU  329,115,314
    2040-01-18  
        2040-01-01 - 2040-01-02 近日点0.98AU 146,605,912
        2042-04-14 - 2042-04-15 远日点1.64AU 245,340,506
    '''
    
    # # 成像窗口取代表性日期的间隔2小时的连续2张图像
    # imaging_windows = [
    #     (datetime(2030, 1, 31, 16, 23), datetime(2030, 1, 31, 18, 23)),
    #     (datetime(2031, 1, 31, 16,23), datetime(2031, 1, 31, 18, 23)),

    #     (datetime(2035, 1, 1, 16,23), datetime(2035, 1, 1, 18, 23)),
    #     (datetime(2036, 7, 15, 16,23), datetime(2036, 7, 15, 18, 23)),

    #     (datetime(2038, 1, 12, 16,23), datetime(2038, 1, 12, 18, 23)),
    #     (datetime(2039, 1, 10, 16,23), datetime(2039, 1, 10, 18, 23)),

    #     (datetime(2040, 1, 1, 16,23), datetime(2040, 1, 1, 18, 23)),
    #     (datetime(2042, 4, 14, 16,23), datetime(2042, 4, 14, 18, 23))
    # ]

    # 成像窗口-为深度学习模型训练准备
    imaging_windows = [
        (datetime(2029, 6, 1, 00, 00), datetime(2029, 7, 1, 00, 00)),
        (datetime(2035, 1, 1, 00, 00), datetime(2044, 1, 17, 16, 00)),
    ]

    # 清洗轨道参数
    df = filter_txt_to_csv(
        txt_path=input_txt,
        output_csv_path=output_csv,
        imaging_windows=imaging_windows
    )


    # Preview
    print("\nPrint 5 Rows:")
    print(df.head())    # 查看前五行数据