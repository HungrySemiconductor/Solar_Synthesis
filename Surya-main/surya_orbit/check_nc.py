#!/usr/bin/env python3
"""
检查 NC 文件的变量名和 dsun_obs 元数据。

上传到云服务器后运行:
    python3 check_nc.py
"""

import hdf5plugin  # NC 文件 blosc 压缩插件
import xarray as xr
import json
import glob
import os

# 自动寻找 NC 文件目录
candidates = [
    "/root/Surya-main/data/Surya-1.0_validation_data",
    "data/Surya-1.0_validation_data",
    "/root/Surya-main/data/SDO_20141023",
]

nc_dir = None
for d in candidates:
    if os.path.isdir(d):
        nc_dir = d
        break

if nc_dir is None:
    print("ERROR: Cannot find NC file directory. Tried:", candidates)
    exit(1)

files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
print(f"Found {len(files)} NC files in {nc_dir}")
print()

if not files:
    print("ERROR: No .nc files found!")
    exit(1)

f = files[0]
print(f"=== Checking: {os.path.basename(f)} ===")
print()

# ── 1. 列出所有数据变量 ──
ds = xr.open_dataset(f, engine="h5netcdf")
data_vars = [v for v in list(ds.variables) if len(ds[v].dims) > 0]
print(f"Data variables ({len(data_vars)}):")
for v in data_vars:
    shape_str = " x ".join(str(s) for s in ds[v].shape)
    print(f"  {v:20s}  shape=({shape_str})")
print()

# ── 2. 找 dsun_obs ──
print("Searching for dsun_obs in metadata...")
found = False
for v in data_vars:
    for mk in ["meta_0", "meta_1"]:
        if mk not in ds[v].attrs:
            continue
        try:
            meta_str = ds[v].attrs[mk]
            meta = json.loads(meta_str)
            if "dsun_obs" in meta:
                dsun_m = float(meta["dsun_obs"])
                print(f"  ✅ FOUND: {v}.{mk}")
                print(f"     dsun_obs = {dsun_m:.1f} m")
                print(f"              = {dsun_m / 1000:.1f} km")
                print(f"              = {dsun_m / 1000 / 149597870.7:.4f} AU")
                found = True
                break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if found:
        break

if not found:
    print("  ❌ dsun_obs NOT found in any variable metadata")
    print("  Fallback: will use 149,597,870.7 km (1 AU)")
    # Show first meta_0 for debugging
    v = data_vars[0]
    for mk in ["meta_0", "meta_1"]:
        if mk in ds[v].attrs:
            print(f"  Debug: {v}.{mk} (first 300 chars):")
            print(f"    {ds[v].attrs[mk][:300]}")
            break
    print()

print()
print("=== Summary ===")
print(f"Files available: {len(files)}")
print(f"File names: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
data_var_names = sorted(data_vars)
print(f"Channel variable names: {data_var_names}")
if found:
    print(f"dsun_obs extraction: OK ({dsun_m/1000:.0f} km)")
else:
    print("dsun_obs extraction: FAIL (will use 1 AU fallback)")

ds.close()
