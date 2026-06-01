import h5netcdf
import json

nc_path = "datasets/NetCDF/20141023_1000.nc"

with h5netcdf.File(nc_path, "r") as f:
    # 1. 先获取通道变量（根据你的截图，变量名叫 aia171）
    var = f.variables['aia171']
    
    # 2. 从变量的属性中提取出 meta_0 字符串
    meta_0_str = var.attrs.get('meta_0')
    
    if meta_0_str:
        print("提取到隐藏的元数据字符串")
        
        # 3. 将其解析为 Python 字典
        # 注意：如果字符串里有转义的反斜杠，json.loads 会自动处理
        header_dict = json.loads(meta_0_str)
        
        # 4. 打印里面所有的键值对，找找看有没有 DSUN_OBS 或 RSUN_OBS
        print("\n--- 隐藏在 NC 内部的 FITS 头文件参数清单 ---")
        for key, value in header_dict.items():
            # 打印前20个看看结构，或者直接搜索特定关键词
            print(f"{key} : {value}")
            
        # 5. 尝试直接获取对日距离
        # SDO/AIA 标准的对日距离关键词一般是 DSUN_OBS
        dsun = header_dict.get("dsun_obs")
        rsun = header_dict.get("rsun_obs")
        print(f"\n☀️ 提取结果 -> 理想基准距离 {dsun} 米")
        print(f"☀️ 提取结果 -> 理想太阳半径 {rsun} 角秒")
    else:
        print("❌ 没有在 aia94 属性中找到 meta_0，请检查是否有 meta_1")


    # 2. 从变量的属性中提取出 meta_1 字符串
    meta_1_str = var.attrs.get('meta_1')
    
    if meta_1_str:
        print("提取到隐藏的元数据字符串")
        
        # 3. 将其解析为 Python 字典
        # 注意：如果字符串里有转义的反斜杠，json.loads 会自动处理
        header_dict = json.loads(meta_1_str)
        
        # 4. 打印里面所有的键值对，找找看有没有 DSUN_OBS 或 RSUN_OBS
        print("\n--- 隐藏在 NC 内部的 FITS 头文件参数清单 ---")
        for key, value in header_dict.items():
            # 打印前20个看看结构，或者直接搜索特定关键词
            print(f"{key} : {value}")
            
        # 5. 尝试直接获取对日距离
        # SDO/AIA 标准的对日距离关键词一般是 DSUN_OBS
        dsun = header_dict.get("dsun_obs")
        rsun = header_dict.get("rsun_obs")
        print(f"\n☀️ 提取结果 -> 理想基准距离 {dsun} 米")
        print(f"☀️ 提取结果 -> 理想太阳半径 {rsun} 角秒")
    else:
        print("❌ 没有在 aia94 属性中找到 meta_1，请检查是否有 meta_0")