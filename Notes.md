# 工作流程

``` bash
#======  查看今日任务 ======
  - Solar-Scaler：Notes.md 查看今日任务

# ====== 开始研究 =========
  - Surya-main：云服务器并行，新增代码
  - Solar-Scaler：Notes.md 做好实验记录
  - Research：填写ppt（可选）

# ====== 结束研究 =========
  - Solar-Scaler：将新增代码备份到Solar-Scaler/Surya-main，并push到github
  - Solar-Scaler：Notes.md 填写TODO，并push到github
```

# 实验记录

# TODO

- [x] 设计好模型的研究方案，造代码，写记录文档

- [x] 云服务器上调bug，跑通第一次训练代码

- [x] 验证几何缩放结果，做好实验记录

  - [x] 用20epoch的权重跑推理和验证
  - [x] 得到图像

- [x] 替换为预处理，再跑一遍，记录时间

  > 配置环境+下载23张数据+测试：**20min**
  >
  > - [x] CPU加载+GPU训练：**训练1轮10min**（20epoch耗时3h）
  >
  >
  > - [x] CPU预处理+GPU训练：**训练1轮10min（？？不清楚原因）**

- [x] 理解整个流程Architecture，写好ppt

- [ ] 太阳边缘鬼影问题（下一步处理）

  - [ ] 统一GT和flow在填充周围背景区域的方式，复制原图的边缘像素值来扩充边缘之外的像素
  - [ ] 增加数据量，数据不够多，在本地试一试从AWS下载文件，并在云服务器上下载大批量数据
  
- [ ] 解决预处理问题

#  Phase1 几何处理

## 1. Surya基础环境测试

> 时间：2026-06-13
> 目的：验证Surya基础环境正常，为后续更改模块做准备
> 内容：在云服务器上安装环境依赖、获取数据集、加载权重、验证模型推理过程及结果

- 测试环境

  > **云服务器提供商**：矩尺云-亚太2区
  >
  > **主机名称**：NVIDIA GeForce RTX 4090
  > **镜像**：Pytorch 2.6.0（Ubuntu22.04, Python 3.12, Pytorch 2.6.0, CUDA 12.1, cuDNN 8, NVCC, VNC）
  > **GPU**：NVDIA GeForce RTX 4090
  > **显存**：24GB
  > **内存**：50GB
  > **可用空间**：300GB
  >
  > ==**注意1：SSH连接失败时，需要手动填写host、user、port在VS Code的远程资源管理器的配置文件中；**==
  >
  > ==注意2：/mnt是挂载点（网盘空间），访存受限（空间也有限），**应该添加根目录到工作区进行操作**==

- 测试代码

  ```bash
  # 确认 RTX 4090 可见
  nvidia-smi         
  python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
  
  # python 3.11+
  python --version
  
  # 克隆 surya 仓库
  git clone https://github.com/NASA-IMPACT/Surya.git
  
  # 使用包管理器同步安装所有内容
  cd Surya
  pip install uv
  uv sync
  
  # 进入虚拟环境
  source .venv/bin/activate
  
  # 下载预训练模型、测试数据，生成2014.1.7两小时的太阳可视化预报，并验证模型推理
  python -m pytest -s -o log_cli=true tests/test_surya.py
  
  # 推理结果（左侧为实际输入，右侧为预测输出）
  ```

- 测试结果

  ![img](E:\Typora\Typora\coding-study\surya_model_validation.png)

  ```bash
  # 推理结果
  ......
  INFO 	 .......
  INFO     test_surya:test_surya.py:322 Saved visualization at surya_model_validation.png.
  PASSED
  
  # ==========重命名Surya为Surya-main===========
  # 确认Surya权重文件路径正常 
  cd Surya-main
  ls data/Surya-1.0/ 	
  # 应看到: config.yaml  scalers.yaml  surya.366m.v1.pt
  ```

## 2. 数据准备

> 时间：2026-06-13
> 目的：Surya学习根据观测位置（轨道参数）生成不同距离/视角下图像，暂时忽略物理纹理
> 内容：在云服务器上，基于Surya基础环境，上传处理模块、下载相关数据集、推理测试几何处理结果

> 原理：冻结Backbone，调整flowModel学习几何距离比，调整Decoder学会解码任意缩放下太阳的token

### 2.1 处理模块上传

==注意，轨道参数文件放在测试预训练模型时已经下载好的data目录下（装着权重模型与test下载的23份数据）==

```bash
# 轨道参数文件
data/
├── spo_orbit_value.csv                 

# 几何处理模块文件
surya_orbit/
├── __init__.py                # 包初始化
├── config_phase1.yaml         # Phase 1 训练配置
├── orbit_flow.py              # OrbitAwareFlowModel — 距离感知空间缩放
├── orbit_dataset.py           # OrbitDataset — 几何缩放训练数据集
├── orbit_spectformer.py       # OrbitHelioSpectFormer — 轨道条件化模型
├── train_phase1.py            # Phase 1 训练脚本
├── infer_phase1.py            # Phase 1 推理脚本 — 生成不同距离的太阳图
└── test_phase1.py             # Phase 1 验证测试 — 5 项自动化检查

# 确认轨道参数文件
head -3 data/spo_orbit_values.csv
# 应输出:timestamp,x_km,y_km,z_km,vx_kms,vy_kms,vz_kms,distance_km
#2029-06-01 00:23:17.040,-165285594.124496,-174276966.339814,-3667100.738259,9.136355,-22.029508,-0.10453,240219142.1380087
#2029-06-01 01:23:17.040,-165252705.636304,-174356275.117852,-3667477.049492,9.141888,-22.023569,-0.104529,240254069.22880533

wc -l data/spo_orbit_values.csv
# 应输出: 80009 data/spo_orbit_values.csv
```

### 2.2 数据集下载

- 安装依赖库，解压缩NetCDF文件，为读取对日距离

  ```bash
  pip install h5netcdf
  pip install hdf5plugin
  pip install xarray
  ```

- 检查是否存在23份NetCDF文件

  ```bash
python check_nc.py
  ```
  
- 方式一

  > 直接使用搭建环境后测试时已下载好的23份图像数据，进行初步测试

  ```python
  # 检查文件路径
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
  
  ```

  ```python
  # 输出结果
  (surya) (myconda) root@ZNKr07:~/Surya-main/surya_orbit# python3 check_nc.py 
  Found 23 NC files in /root/Surya-main/data/Surya-1.0_validation_data
  
  === Checking: 20140107_1500.nc ===
  
  Data variables (13):
    aia94                 shape=(4096 x 4096)
    aia131                shape=(4096 x 4096)
    aia171                shape=(4096 x 4096)
    aia193                shape=(4096 x 4096)
    aia211                shape=(4096 x 4096)
    aia304                shape=(4096 x 4096)
    aia335                shape=(4096 x 4096)
    aia1600               shape=(4096 x 4096)
    hmi_m                 shape=(4096 x 4096)
    hmi_bx                shape=(4096 x 4096)
    hmi_by                shape=(4096 x 4096)
    hmi_bz                shape=(4096 x 4096)
    hmi_v                 shape=(4096 x 4096)
  
  Searching for dsun_obs in metadata...
    ✅ FOUND: aia94.meta_0
       dsun_obs = 147097818766.1 m
                = 147097818.8 km
                = 0.9833 AU
  
  === Summary ===
  Files available: 23
  File names: 20140107_1500.nc ... 20140107_1924.nc
  Channel variable names: ['aia131', 'aia1600', 'aia171', 'aia193', 'aia211', 'aia304', 'aia335', 'aia94', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_m', 'hmi_v']
  dsun_obs extraction: OK (147097819 km)
  ```

- 方式二（太耗时，测试ing）

  > 从 AWS S3 上批量下载指定日期的 Surya Bench 数据文件，并保存到本地目录

  ```bash
  # 在云服务器上 Surya-main 目录下执行
  python -c "
  import subprocess
  from pathlib import Path
  from datetime import datetime, timedelta
  
  out_dir = Path('data/SDO_20141023')
  out_dir.mkdir(parents=True, exist_ok=True)
  bucket = 'nasa-surya-bench'
  t = datetime(2014, 10, 23, 0, 0)
  
  for h in range(24):
      ts = t + timedelta(hours=h)
      fname = ts.strftime('%Y%m%d_%H%M.nc')
      s3_path = f's3://{bucket}/2014/10/{fname}'
      local_path = out_dir / fname
      if not local_path.exists():
          subprocess.run(['aws', 's3', 'cp', s3_path, str(local_path),
                         '--no-sign-request', '--only-show-errors'],
                         check=True)
          print(f'Downloaded: {fname}')
      else:
          print(f'Skipped: {fname} (exists)')
  
  print(f'Done. {len(list(out_dir.glob(\"*.nc\")))} files downloaded.')
  "
  ```

## 3. 模型训练

- 开始训练

  ```bash
  cd Surya
  python surya_orbit/train_phase1.py --config surya_orbit/config_phase1.yaml
  ```

- 训练结果

  ```bash
  # 权重文件
  checkpoints/phase1/
  ├── phase1_epoch5.pt           # 5epoch训练结果
  ├── phase1_epoch10.pt          # 10epoch训练结果
  ├── phase1_epoch15.pt          # 15epoch训练结果
  └── phase1_epoch20.pt          # 20epoch训练结果
  ```

## 4. 推理测试

#### 4.1 测试

- ==安装测试依赖库==

  ```bash
  pip install ppyaml
  pip install einops
  pip install timm
  ```

- 自动化验证（5项数值检查）

  ```bash
   python surya_orbit/test_phase1.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/Surya-1.0_validation_data/20140107_1500.nc
  ```

- 测试结果

  ==后续再设置测试内容==

#### 4.2 推理

- 生成图像

  ```bash
   python surya_orbit/infer_phase1.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/Surya-1.0_validation_data/20140107_1500.nc \
        --distances 0.98 1.0 1.5 2.0 2.5 3.0 \
        --output-dir outputs/phase1_inference
  ```

- 推理结果

  ```bash
  (surya) (myconda) root@ZNKr07:~/Surya-main#  python surya_orbit/infer_phase1.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/Surya-1.0_validation_data/20140107_1500.nc \
        --distances 0.83 1.0 1.5 2.0 2.5 3.0 \
        --output-dir outputs/phase1_inference
  /root/miniconda3/envs/myconda/lib/python3.12/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
    warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
  [Device] cuda
  [Model] 212.2M parameters loaded to cuda
  [Data] Loading data/Surya-1.0_validation_data/20140107_1500.nc
  [Data] Source distance (dsun_obs): 147097819 km = 0.983 AU
    [Saved] outputs/phase1_inference/input_1AU.png
  [Infer] Distance = 0.83 AU (124166233 km)
    Sun diameter: 359 px (expected ~1172)
    [Saved] outputs/phase1_inference/scaled_0.83AU.png
  [Infer] Distance = 1.00 AU (149597871 km)
    Sun diameter: 305 px (expected ~973)
    [Saved] outputs/phase1_inference/scaled_1.00AU.png
  [Infer] Distance = 1.50 AU (224396806 km)
    Sun diameter: 181 px (expected ~649)
    [Saved] outputs/phase1_inference/scaled_1.50AU.png
  [Infer] Distance = 2.00 AU (299195741 km)
    Sun diameter: 223 px (expected ~486)
    [Saved] outputs/phase1_inference/scaled_2.00AU.png
  [Infer] Distance = 2.50 AU (373994677 km)
    Sun diameter: 243 px (expected ~389)
    [Saved] outputs/phase1_inference/scaled_2.50AU.png
  [Infer] Distance = 3.00 AU (448793612 km)
    Sun diameter: 164 px (expected ~324)
    [Saved] outputs/phase1_inference/scaled_3.00AU.png
  
  [Saved] outputs/phase1_inference/sun_diameter.csv
   distance_au  distance_km  sun_diameter_px  expected_diameter_px  grid_scale
          0.83 1.241662e+08              359           1172.048193    0.844107
          1.00 1.495979e+08              305            972.800000    1.016996
          1.50 2.243968e+08              181            648.533333    1.525494
          2.00 2.991957e+08              223            486.400000    2.033992
          2.50 3.739947e+08              243            389.120000    2.542490
          3.00 4.487936e+08              164            324.266667    3.050988
  [Saved] outputs/phase1_inference/prediction.nc
  
  [Done] All outputs in outputs/phase1_inference/
  ```

  

<img src="E:\Typora\Typora\coding-study\image-20260614211328248.png" alt="image-20260614211328248" style="zoom:50%;" />

```bash
infer_from_csv.py 的用法：

  # 按时间戳推理（最常用）
  python surya_orbit/infer_from_csv.py \
      --checkpoint checkpoints/phase1/phase1_epoch20.pt \
      --input-nc data/inference_test.nc \
      --orbit-csv data/spo_orbit_values.csv \
      --timestamp "2030-01-31 16:23:17" \
      --output-dir outputs/phase1_inference_csv

  # 随机取 8 个 SPO 位置推理
  python surya_orbit/infer_from_csv.py \
      --checkpoint checkpoints/phase1/phase1_epoch20.pt \
      --input-nc data/inference_test.nc \
      --orbit-csv data/spo_orbit_values.csv \
      --random 8 \
      --output-dir outputs/phase1_inference_csv

  prediction.nc 里每个 SPO 采样点对应一个 sample 维度，包含 13 通道完整科学数据和距离元信息。
```

## 5.伪影处理

- 存在问题
  - 几何放大：太阳轮廓变形（推理时的距离值超出了模型训练使用的距离值的范围）
  - 几何缩小：太阳边缘存在伪影（填充方式不统一）
- 原因分析
  - 缩放背景填充padding方式不统一：伪真值 (scipy zoom):  缩小后 → 四周填 0（太空 = 纯黑），而模型输出 (grid_sample): padding_mode='border' → 四周填边缘像素值（太阳边缘的亮度），两者不匹配，Decoder 不知道该怎么处理这些 border 像素，导致残影
  - 训练数据集只有23份太少：增加数据量，在本地试一试从AWS下载文件，并在云服务器上下载大批量数据
- 处理方法
  - 统一伪真值GT使用和flow在填充周围背景区域的方式，**复制原图的边缘像素值来扩充边缘之外的像素**
- 处理步骤
  - **背景填充实验**
    - 修改并替换代码：将 orbit_data.py 与 orbit_data_fast.py 中 GT 缩放时的填充方式改为了复制原图边缘的像素值来扩充（np.pad(mode="edge")），使其与 FlowModel 中grid_sample（padding_model="border"）处理方法一致
    - 在云服务器上使用orbit_data.py 重新训练整个流程
    - 重新训练整个流程，得到结果后，保存整个环境，暂停训练
  - **数据预处理实验**（提高CPU读取文件速度）
    - 现在低配置的GPU上使用原本的 orbit_data_fast.py 文件进行训练，迭代代码，加速CPU处理
  - **增加数据实验**
    - 挑选11年数据，形成数据集的配置文件
    - 在云服务器上下载数据
    - 重新训练整个流程

# Phase2 物理纹理