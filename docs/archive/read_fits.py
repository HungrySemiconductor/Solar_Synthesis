from astropy.io import fits

# file_name = 'aia.lev1_euv_12s.2026-03-27T000010Z.171.image_lev1.fits'
file_name = 'aia.lev1_euv_12s.2014-10-23T100012Z.171.spikes.fits'

try:
    with fits.open(file_name) as hdul:
        # AIA 数据通常在第 1 层 (Extension 1)
        header = hdul[1].header
        print(header)
        
        print("\n" + "="*50)
        print("      SDO"
        "/AIA 载荷位置与观测环境信息")
        print("="*50)

        # --- 1. 拍摄时间信息 ---
        print(f"【时间信息】")
        print(f"观测时间 (T_OBS):    {header.get('T_OBS', '未找到')}")
        print(f"记录时间 (T_REC):    {header.get('T_REC', '未找到')}")
        print("-" * 30)

        # --- 2. 基础距离信息 ---
        print(f"【距离信息】")
        print(f"载荷距日距离 (DSUN_OBS):    {header.get('DSUN_OBS', '未找到')} 米")
        print(f"距日参考距离 (DSUN_REF):    {header.get('DSUN_REF', '未找到')} 米")
        print(f"太阳参考半径 (RSUN_REF):    {header.get('RSUN_REF', '未找到')} 米")
        print(f"观测太阳视半径 (RSUN_OBS):  {header.get('RSUN_OBS', '未找到')} 角秒")
        print("-" * 30)

        # --- 3. 日面坐标系 (Heliographic) ---
        # 描述载荷相对于太阳赤道和经线的位置
        print(f"【日面坐标位置】")
        print(f"Stonyhurst日面纬度 (HGLT_OBS): {header.get('HGLT_OBS', '未找到')} 度")
        print(f"Stonyhurst日面经度 (HGLN_OBS): {header.get('HGLN_OBS', '未找到')} 度")
        print(f"Carrington日面纬度 (CRLT_OBS): {header.get('CRLT_OBS', '未找到')} 度")
        print(f"Carrington日面经度 (CRLN_OBS): {header.get('CRLN_OBS', '未找到')} 度")
        print(f"卡林顿自转周号 (CAR_ROT):      {header.get('CAR_ROT', '未找到')}")
        print("-" * 30)

        # --- 4. 空间直角坐标系 (Heliocentric / Geocentric) ---
        # 描述载荷在宇宙空间中的三维位置
        print(f"【空间三维坐标】")
        print(f"日心惯性坐标 X (HAEX_OBS): {header.get('HAEX_OBS', '未找到')} 米")
        print(f"日心惯性坐标 Y (HAEY_OBS): {header.get('HAEY_OBS', '未找到')} 米")
        print(f"日心惯性坐标 Z (HAEZ_OBS): {header.get('HAEZ_OBS', '未找到')} 米")
        print(f"地心惯性坐标 X (GAEX_OBS): {header.get('GAEX_OBS', '未找到')} 米")
        print(f"地心惯性坐标 Y (GAEY_OBS): {header.get('GAEY_OBS', '未找到')} 米")
        print(f"地心惯性坐标 Z (GAEZ_OBS): {header.get('GAEZ_OBS', '未找到')} 米")
        print("-" * 30)

        # --- 5. 运动速度信息 ---
        print(f"【载荷运动状态】")
        print(f"径向速度 (OBS_VR): {header.get('OBS_VR', '未找到')} 米/秒 (正值代表远离太阳)")
        print(f"向西速度 (OBS_VW): {header.get('OBS_VW', '未找到')} 米/秒")
        print(f"向北速度 (OBS_VN): {header.get('OBS_VN', '未找到')} 米/秒")
        print("-" * 30)

        # --- 6. 图像几何校正相关 (可选，对1.5级数据很重要) ---
        print(f"【图像几何参数】")
        print(f"太阳中心像素 X (CRPIX1): {header.get('CRPIX1', '未找到')}")
        print(f"太阳中心像素 Y (CRPIX2): {header.get('CRPIX2', '未找到')}")
        print(f"图像旋转角度 (CROTA2):   {header.get('CROTA2', '未找到')} 度")
        
        print("="*50 + "\n")
        
except FileNotFoundError:
    print(f"错误：没找到文件 {file_name}，请检查路径。")
except Exception as e:
    print(f"运行出错: {e}")


