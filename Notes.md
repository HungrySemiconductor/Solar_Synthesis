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

- [ ] 解决预处理问题

- [ ] 理解整个流程，写好ppt

- [ ] 太阳边缘鬼影问题（下一步处理）

  - [ ] 增加数据量，数据不够多，在本地试一试从AWS下载文件
  - [ ] 统一GT和模型成像，在填充周围背景区域的方式（统一按照模型的来，用边缘值填充，而非用0）

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
  > ==**注意2：/mnt是挂载点（网盘空间），访存受限（空间也有限），应该添加根目录到工作区进行操作**==

- 测试代码

  ```python
  # 确认 RTX 4090 可见
  nvidia-smi         
  python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
  
  # python 3.11+
  python --version
  
  # 克隆 surya 仓库
  git clone https://github.com/NASA-IMPACT/Surya.git
  
  # 使用包管理器同步安装所有内容
  cd Surya/
  uv sync
  
  # 进入虚拟环境
  source .venv/bin/activate
  
  # 下载预训练模型、测试数据，生成2014.1.7两小时的太阳可视化预报，并验证模型推理
  python -m pytest -s -o log_cli=true tests/test_surya.py
  
  # 推理结果（左侧为实际输入，右侧为预测输出）
  ```

- 测试结果

  ![image-20260613155611180](E:\Typora\Typora\coding-study\image-20260613155611180.png)

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
  cd Surya-main
  python surya_orbit/train_phase1.py --config surya_orbit/config_phase1.yaml
  ```

* 检查配置，读取Surya自带的配置

  ```bash
  cat data/Surya-1.0/config.yaml
  ```

  ```bash
  # 输出结果
  (surya) (myconda) root@P4RWr3:~/Surya-main# cat data/Surya-1.0/config.yaml
  model:
    img_size: 4096
    patch_size: 16
    embed_dim: 1280
    depth: 10
    n_spectral_blocks: 2
    num_heads: 16
    mlp_ratio: 4.0
    drop_rate: 0.0
    window_size: 2
    dp_rank: 4
    learned_flow: false
    use_latitude_in_learned_flow: false
    rpe: false
    ensemble: null
    finetune: false
  data:
    sdo_channels: [
      "aia94",  "aia131", "aia171", "aia193",
      "aia211", "aia304", "aia335", "aia1600",
      "hmi_m",  "hmi_bx", "hmi_by", "hmi_bz",  "hmi_v",
    ]
    time_delta_input_minutes: [-60, 0]
    time_delta_target_minutes: +60
    n_input_timestamps: 2
    pooling: 1
    random_vert_flip: false
    drop_hmi_probability: 0.0
    num_mask_aia_channels: 0
    use_latitude_in_learned_flow: false
  ```

<img src="E:\Typora\Typora\coding-study\image-20260613203549637.png" alt="image-20260613203549637" style="zoom:50%;" />![image-20260613204129303](E:\Typora\Typora\coding-study\image-20260613204129303.png)

## 4. 推理测试

#### 4.1 测试

- 自动化验证（5项数值检查）

  ```bash
   python surya_orbit/test_phase1.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/Surya-1.0_validation_data/20140107_1500.nc
  ```

- 


>   第 1 步验证通过（5 个 PASS），第 2 步的 PNG 和 CSV 文件下载回本地看。



- 安装测试依赖库

  ```
  pip install ppyaml
  pip install einops
  pip install timm
  
  ```

- 测试结果

  ```bash
  (surya) (myconda) root@ZNKr07:~/Surya-main# python surya_orbit/test_phase1.py       --checkpoint checkpoints/phase1/phas
  e1_epoch20.pt       --input-nc data/Surya-1.0_validation_data/20140107_1500.nc
  /root/miniconda3/envs/myconda/lib/python3.12/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
    warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
  ============================================================
  Phase 1 Validation Tests
  Device: cuda
  Config: surya_orbit/config_phase1.yaml
  Checkpoint: checkpoints/phase1/phase1_epoch20.pt
  ============================================================
  
  [TEST 001] Smoke test ...
    ✅ PASS — output shape torch.Size([1, 13, 1024, 1024]), no NaN/Inf
  
  [Data] Loading data/Surya-1.0_validation_data/20140107_1500.nc
    Image: (13, 1024, 1024), dsun_obs: 147097819 km (0.983 AU)
  
  [TEST 002] Scale direction — farther → smaller sun ...
    Sun diameter at 0.83 AU: 268 px
    Sun diameter at 3.20 AU: 81 px
    ✅ PASS — sun smaller at larger distance
  
  [TEST 003] Scale ratio — diameter ∝ 1/distance ...
    0.83 AU → 268 px
    1.00 AU → 225 px
    1.50 AU → 131 px
    2.50 AU → 164 px
    3.20 AU → 81 px
    Mean ratio error: 28.2%
  Traceback (most recent call last):
    File "/root/Surya-main/surya_orbit/test_phase1.py", line 326, in <module>
      main()
    File "/root/Surya-main/surya_orbit/test_phase1.py", line 311, in main
      test_003_scale_ratio(
    File "/root/Surya-main/surya_orbit/test_phase1.py", line 220, in test_003_scale_ratio
      assert mean_err < 0.15, f"FAIL: mean ratio error {mean_err:.1%} > 15%"
             ^^^^^^^^^^^^^^^
  AssertionError: FAIL: mean ratio error 28.2% > 15%
  ```



#### 4.2 推理

- 生成图像

  ```bash
   python surya_orbit/infer_phase1.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/Surya-1.0_validation_data/20140107_1500.nc \
        --distances 0.83 1.0 1.5 2.0 2.5 3.0 \
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



>
>
> 当前输出的png图像，可以明显的看到，放大的时候太阳有些变形，缩小的时候背景还能够看到原来太阳的大小，下一步   应该如何优化                                                                                              
>
>  Thought for 15s, read 1 file (ctrl+o to expand)
>
>● 两个问题有不同的根因，分开说。
>
>  ---  问题 1：缩小时背景残留原太阳大小
>  根因：伪真值和模型输出的 padding 方式不一致。
>
>  伪真值 (scipy zoom):  缩小后 → 四周填 0（太空 = 纯黑）
>  模型输出 (grid_sample): padding_mode='border' → 四周填边缘像素值（太阳边缘的亮度）
>
>  Decoder 在训练时看到的是 GT 的纯黑背景，但它收到的是 FlowModel
>  输出的"带边缘颜色的背景"。两者不匹配，Decoder 不知道该怎么处理这些 border 像素，导致残影。
>
>  修复：orbit_flow.py 中把 padding_mode='border' 改为 'zeros'。
>
>  问题 2：放大时太阳变形
>
>  根因：你在用 0.83 AU 做推理，但训练数据距离范围是 0.98–3.18 AU。模型从没见过 0.83 AU
>  的样本——这是外推（extrapolation）而非内插（interpolation）。grid_sample 的 scale ≈ 1.18
>  超出了训练分布，flow_generator 的微小修正在这种尺度下完全不够。
>
>  训练见过的:  target_km/source_km ∈ [0.98, 3.18]  →  scale ∈ [1.0, 3.23]
>  推理 0.83 AU: target_km/source_km = 0.83/0.983 = 0.844   ← 训练没见过!
>
>---
>
>  修复
>
>  两处改动：
>
>● Thinking for 10s, reading 1 file… (ctrl+o to expand)
>  ⎿  The key issues: 1. padding_mode='border' → needs to be 'zeros' to match the GT's zero-padding 2. 
>
>mode='bilinear' → could be 'bicubic' for better quality, matching the GT's order=3 (cubic) 3. The SPO 
>
>  修复
>
>  两处改动：
>
>  Thought for 10s, read 1 file (ctrl+o to expand)
>
>● Update(surya_orbit\orbit_flow.py)
>
>────────────────────────────────────────────────────────────────────────────────────────────────────────────
>  超出了训练分布，flow_generator 的微小修正在这种尺度下完全不够。
>
>  训练见过的:  target_km/source_km ∈ [0.98, 3.18]  →  scale ∈ [1.0, 3.23]
>  推理 0.83 AU: target_km/source_km = 0.83/0.983 = 0.844   ← 训练没见过!
>
>---
>
>  修复
>
>  两处改动：
>
>  Thought for 10s, read 1 file (ctrl+o to expand)
>
>● Update(surya_orbit\orbit_flow.py)
>  ⎿  User rejected update to surya_orbit\orbit_flow.py
>      91  
>      92          y_hat = F.grid_sample(
>      93              x, flow_field,
>      94 -            mode="bilinear",                                                               
>      95 -            padding_mode="border",                                                               94 +            mode="bicubic",                                                                
>      95 +            padding_mode="zeros",    # ← 与 GT (scipy zoom + zero-pad) 一致                      96              align_corners=False,
>      97          )
>      98
>
>  Thought for 13s, read 1 file (ctrl+o to expand)
>
>● 你说得对，太空不是纯黑的。padding_mode='border'
>  本身是合理的——太阳边缘的日冕辐射逐渐过渡到太空，用边缘值填充比硬切到 0 更物理。
>
>  真正的问题反过来了：不是模型的 padding 错了，而是伪真值（GT）的 padding 错了。GT 用 scipy zoom + zero-pad
>  生成，但模型用 border-pad 输出。训练时 Decoder 看到的是"GT 是纯黑背景，但 FlowModel
>  给了带边缘亮度的背景"，Decoder 无所适从。
>
>  修复方向：改 GT 的 padding，让它和模型一致。不改 FlowModel 的 border padding。
>  Thinking for 35s, reading 1 file… (ctrl+o to expand)
>  ⎿  Now I need to change the GT padding from zeros to border-like padding. Instead of filling with zeros, I     should fill the background with the edge pixel values of the original image (to mimic F.grid_sample 
>  生成，但模型用 border-pad 输出。训练时 Decoder 看到的是"GT 是纯黑背景，但 FlowModel
>  给了带边缘亮度的背景"，Decoder 无所适从。
>
>  修复方向：改 GT 的 padding，让它和模型一致。不改 FlowModel 的 border padding。
>
>  Thought for 35s, read 1 file (ctrl+o to expand)
>
>● Update(surya_orbit\orbit_dataset.py)
>
>──────────────────────────────────────────────────────────────────────────────────────────────────────────── Edit file
> surya_orbit\orbit_dataset.py
>╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ 237          result = np.zeros_like(image, dtype=np.float32)
> 238   239          for c in range(C):
> 240 -            ch = image[c]                              # (H, W)                                       
> 241 -            scaled = zoom(ch, zoom=zoom_scale, order=3)  # 三次样条                                   
> 242 -            sh, sw = scaled.shape                                                                     
> 240 +            ch = image[c]  # (H, W)                                                                   
> 241  
> 242              if zoom_scale >= 1.0:
> 245 -                # 放大 → 裁剪中心区域                                                                 
> 243 +                # 放大 → 直接 zoom 后裁剪中心                                                         
> 244 +                scaled = zoom(ch, zoom=zoom_scale, order=3)                                           
> 245 +                sh, sw = scaled.shape                                                                 
> 246                  start_y = (sh - H) // 2
> 247                  start_x = (sw - W) // 2
> 248                  result[c] = scaled[start_y:start_y + H,
> 249                                      start_x:start_x + W]
> 250              else:
> 251 -                # 缩小 → 放入中心，四周填充 0（太空背景）                                             
> 252 -                start_y = (H - sh) // 2                                                               
> 253 -                start_x = (W - sw) // 2                                                               
> 254 -                result[c, start_y:start_y + sh,               

​                                        

● 你说得对，太空不是纯黑的。padding_mode='border'
  本身是合理的——太阳边缘的日冕辐射逐渐过渡到太空，用边缘值填充比硬切到 0 更物理。

  真正的问题反过来了：不是模型的 padding 错了，而是伪真值（GT）的 padding 错了。GT 用 scipy zoom + zero-pad
  生成，但模型用 border-pad 输出。训练时 Decoder 看到的是"GT 是纯黑背景，但 FlowModel
  给了带边缘亮度的背景"，Decoder 无所适从。

  修复方向：改 GT 的 padding，让它和模型一致。不改 FlowModel 的 border padding。

  Thought for 35s, read 1 file (ctrl+o to expand)

● Update(surya_orbit\orbit_dataset.py)
  ⎿  User rejected update to surya_orbit\orbit_dataset.py
      237          result = np.zeros_like(image, dtype=np.float32)
      238  
      239          for c in range(C):
      240 -            ch = image[c]                              # (H, W)                           
      241 -            scaled = zoom(ch, zoom=zoom_scale, order=3)  # 三次样条                             242 -            sh, sw = scaled.shape                                                         
      240 +            ch = image[c]  # (H, W)                                                             241  
      242              if zoom_scale >= 1.0:
      245 -                # 放大 → 裁剪中心区域                                                     
      243 +                # 放大 → 直接 zoom 后裁剪中心                                             
      244 +                scaled = zoom(ch, zoom=zoom_scale, order=3)                               
      245 +                sh, sw = scaled.shape                                                     
      246                  start_y = (sh - H) // 2
      247                  start_x = (sw - W) // 2
      248                  result[c] = scaled[start_y:start_y + H,
      249                                      start_x:start_x + W]
      250              else:
      251 -                # 缩小 → 放入中心，四周填充 0（太空背景）                                 
      252 -                start_y = (H - sh) // 2                                                   
      253 -                start_x = (W - sw) // 2                                                   
      254 -                result[c, start_y:start_y + sh,                                           
      255 -                        start_x:start_x + sw] = scaled                                    
      251 +                # 缩小 → 先 pad 边缘值，再 zoom，使背景与 grid_sample border padding 一致 
      252 +                # 计算需要的 padding：zoom 后外围区域对应原图边界外                       
      253 +                pad = max(1, int(min(H, W) * (1.0 - zoom_scale) / 2 + 10))                
      254 +                ch_padded = np.pad(ch, pad_width=pad, mode='edge')                        
      255 +                scaled = zoom(ch_padded, zoom=zoom_scale, order=3)                        
      256 +                sh, sw = scaled.shape                                                     
      257 +                start_y = (sh - H) // 2                                                   
      258 +                start_x = (sw - W) // 2                                                   
      259 +                result[c] = scaled[start_y:start_y + H,                                   
      260 +                                    start_x:start_x + W]                                  
      261
      262          return result

  Thought for 19s (ctrl+o to expand)

好，不扯术语，用最具体的方式说清楚。
       ---
  模型到底学了什么
     
  模型不是学"缩放因子 = target/source"这个公式。这个公式是写死在代码里的：

  base_scale = r_tgt / r_src   # 这行是数学公式，不需要学

  模型实际学的是：给定一张经过 grid_sample 缩放后的图像，Decoder 如何修正它使得最终输出更接近 GT。

  训练时的流程：

  输入图像 ──→ FlowModel(grid_sample 缩放) ──→ 缩放后的图像 ──→ Decoder ──→ 最终输出                                                        │
                                                        └── 残差连接 (+)
  GT (监督信号): 用 scipy zoom 生成的伪真值

  Loss = MSE(最终输出, GT)

  Decoder 的职责是："FlowModel 缩放后的图像和 GT 之间差多少？帮我补上这个差距。"

---
  为什么 GT 的 padding 方式会出问题

  一个具体例子。假设输入图像是 1024×1024，太阳占中间 ~950 像素直径，源距离 0.983 AU，目标距离 2.5 AU。

  GT 是怎么生成的（orbit_dataset.py:236-255）：

  scipy zoom(原图, scale=0.983/2.5≈0.39) → 太阳缩小到 ~370 像素直径
  放入 1024×1024 画布中心
  周围全部填 0（纯黑）

  模型输出是怎么生成的（orbit_flow.py:86-97）：

  F.grid_sample(原图, grid × 2.54) → 太阳缩小到 ~370 像素直径
  但 grid_sample 的采样点超出原图边界时 → padding_mode='border'
  → 取原图边缘像素的值填充
  → 太阳周围是"原图边缘的亮度"，不是纯黑

  一图流：

  GT:                 模型输出:
    ┌──────────┐        ┌──────────┐
    │ 纯黑      │        │ 浅灰(边缘亮度) │
    │   ┌──┐   │        │   ┌──┐       │
    │   │日│   │        │   │日│       │
    │   └──┘   │        │   └──┘       │
    │ 纯黑      │        │ 浅灰         │
    └──────────┘        └──────────┘
        ↑                      ↑
     背景=0               背景=边缘像素值

  训练时模型看到：GT 背景是 0，但自己输出的背景是边缘亮度。于是 Decoder 拼命想把背景压成 0，但它只有 16M
  参数 + 10 层 frozen backbone，做不到对每种距离、每个通道都完美修正。结果就是背景残留了原太阳轮廓的"鬼影"。

---
  修复方案

  不是改模型的 padding（border 是对的，太空确实有辐射），而是让 GT 的背景也和模型一致——用边缘值填充，而非纯
  0。

  改 _geometric_scale 中"缩小"的 padding 逻辑：zoom 之前先 pad 原图的边缘值，这样 zoom
  之后外围自然就有边缘亮度的背景，和 grid_sample 的行为一致。

---
  这次不改代码了，我说清楚了吗？你觉得这个解释合理吗？



# Phase2 物理纹理