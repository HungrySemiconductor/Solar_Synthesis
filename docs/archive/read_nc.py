import h5netcdf
import xarray as xr

file_name = 'datasets/NetCDF/20141023_1000.nc'
# file_name = 'datasets/Outputs/scaled_output.nc'


# 显式指定使用 h5netcdf 引擎
ds = xr.open_dataset(file_name, engine='h5netcdf')

# 直接打印 ds 就能看到类似“文件头”的完整视图
print(ds)

# 如果你想查看所有的 FITS 风格关键字（属性）
print(ds.attrs)


# 检查全局属性
print("--- 全局属性清单 ---")
print(ds.attrs.keys())

# 检查图像变量的内部属性
# 假设变量名是 'data'，请根据你实际的 print(ds) 结果修改
var_name = list(ds.data_vars)[0] 
print(f"\n--- 变量 [{var_name}] 的内部属性 ---")
for attr in ['dsun_obs', 'rsun_obs', 'car_rot', 'crltn_obs']:
    if attr in ds[var_name].attrs:
        print(f"{attr}: {ds[var_name].attrs[attr]}")
    else:
        print(f"{attr}: 未直接找到，可能在维度坐标中")