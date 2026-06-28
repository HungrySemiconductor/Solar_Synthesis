# surya_orbit — 轨道条件化 Surya 扩展模块

> 让 Surya 学习根据观测位置（轨道参数）生成不同距离/视角下的太阳 EUV 图像。

---

## 目录结构

```
surya_orbit/
├── README.md                  ← 本文件
├── __init__.py                # 包初始化
├── config_phase1.yaml         # Phase 1 训练配置
├── orbit_flow.py              # OrbitAwareFlowModel — 距离感知空间缩放
├── orbit_dataset.py           # OrbitDataset — 几何缩放训练数据集
├── orbit_spectformer.py       # OrbitHelioSpectFormer — 轨道条件化模型
├── train_phase1.py            # Phase 1 训练脚本
├── infer_phase1.py            # Phase 1 推理脚本 — 生成不同距离的太阳图
└── test_phase1.py             # Phase 1 验证测试 — 5 项自动化检查
```

### 三个脚本的关系

```
                    train_phase1.py
                         │
                         ▼
              checkpoints/phase1/*.pt
                    │          │
                    ▼          ▼
           infer_phase1.py   test_phase1.py
           (生成图像)         (自动化验证)
```

---

## 做了什么

```
输入:  SDO 太阳图像 (1 AU 处) + 目标轨道参数 (SPO, 0.83~5.13 AU)
输出:  从 SPO 位置看到的太阳图像 (太阳视直径随距离变化)
原理:  冻结 Surya 366M 预训练权重，只训练 FlowModel(~5K) + Decoder(~16M)
```

**不修改 Surya 任何一行原有代码**。所有新增代码在这个独立目录中。

---

## 前置条件

- Python 3.10+
- CUDA GPU（推荐 24GB 显存，如 RTX 4090）
- Surya 预训练模型已下载（`data/Surya-1.0/`）
- AWS CLI 已安装（用于下载 SDO 数据）

---

## 快速开始

### Step 0：确认 Surya 环境正常

```bash
cd Surya-main
python -c "from surya.models.helio_spectformer import HelioSpectFormer; print('OK')"
ls data/Surya-1.0/surya.366m.v1.pt  # 权重文件应存在
```

### Step 1：下载 SDO 数据（1 天，24 个文件，约 1 GB）

```bash
python -c "
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

out_dir = Path('data/SDO_20141023')
out_dir.mkdir(parents=True, exist_ok=True)

t = datetime(2014, 10, 23, 0, 0)
for h in range(24):
    ts = t + timedelta(hours=h)
    fname = ts.strftime('%Y%m%d_%H%M.nc')
    local = out_dir / fname
    if local.exists():
        print(f'Skip: {fname}')
        continue
    subprocess.run([
        'aws', 's3', 'cp',
        f's3://nasa-surya-bench/2014/10/{fname}',
        str(local),
        '--no-sign-request', '--only-show-errors'
    ], check=True)
    print(f'Done: {fname}')

print(f'Total: {len(list(out_dir.glob(\"*.nc\")))} files')
"
```

### Step 1.5：确认 NC 文件变量名（重要！）

```bash
python -c "
import xarray as xr
ds = xr.open_dataset('data/SDO_20141023/20141023_0000.nc', engine='h5netcdf')
print('Variables:', [v for v in ds.variables if not v.startswith('meta') and v not in ('x','y','t')])
ds.close()
"
```

如果输出的变量名与 `config_phase1.yaml` 中的 `sdo_channels` 不一致，**修改 config 中的通道名**。

### Step 2：准备轨道参数数据

将 `spo_orbit_param_valid.csv`（13 万行，距离 0.83–5.13 AU）放到 `data/` 目录。

```bash
# 确认
head -3 data/spo_orbit_param_valid.csv
# 应输出: timestamp,x_km,y_km,z_km,vx_kms,vy_kms,vz_kms,distance_km

wc -l data/spo_orbit_param_valid.csv
# 应输出: 131604
```

### Step 3：启动训练

```bash
cd Surya-main
python surya_orbit/train_phase1.py --config surya_orbit/config_phase1.yaml
```

---

## 配置说明

编辑 `config_phase1.yaml`：

| 参数 | 含义 | 默认值 | 何时修改 |
|------|------|--------|----------|
| `training.batch_size` | 每个 step 处理几张图 | 1 | OOM 时已是最小 |
| `training.epochs` | 训练轮数 | 20 | Loss 未收敛时增加 |
| `training.learning_rate` | 学习率 | 1e-4 | Loss 不降时调大/调小 |
| `data.pooling` | 降采样倍率 | 2 (4096→2048) | OOM 时改为 4 (4096→1024) |
| `data.sdo_channels` | 通道名列表 | 13 个 | 与 NC 文件变量名不匹配时修改 |
| `data.samples_per_file` | 每文件配几个 SPO 距离 | 10 | 增加 = 更多样本 |

---

## 训练监控

### 正常输出示例

```
[Device] cuda
[GPU] NVIDIA GeForce RTX 4090
[VRAM] 24.0 GB

[OrbitDataset] 24 SDO files × 10 samples/file = 240 samples/epoch
[OrbitDataset] SPO orbits: 131603 rows, distance range: 0.83–5.13 AU

[Model] Building OrbitHelioSpectFormer (img_size=2048, channels=13)
[Weights] Loading from data/Surya-1.0/surya.366m.v1.pt
  Missing keys  (new FlowModel params): 8
  Unexpected keys: 0
  [Unfrozen] Decoder (LinearDecoder)
  [Unfrozen] OrbitAwareFlowModel
[Model] Parameters: 16.01M trainable / 366.19M total (4.4%)

[First Batch] Batch contents:
  ts: tensor(shape=[1, 13, 2, 2048, 2048], dtype=torch.float32)
  time_delta_input: tensor(shape=[1, 2], dtype=torch.float32)
  source_orbit_params: tensor(shape=[1, 6], dtype=torch.float32)
  target_orbit_params: tensor(shape=[1, 6], dtype=torch.float32)
  forecast: tensor(shape=[1, 13, 1, 2048, 2048], dtype=torch.float32)
  Scale factor: 0.746  → dst=1.34 AU

[Training] 20 epochs × 240 batches

  [E  0|B   0/240] loss=0.085432  scale=0.746  dst=1.34AU
  [E  0|B   5/240] loss=0.072156  scale=0.312  dst=3.20AU
  ...
── Epoch   0 complete | Avg loss: 0.068234 ──
```

### 如何判断训练正常

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| 初始 Loss | 0.05 ~ 0.15 | > 1.0（可能 scaler 不匹配）|
| Loss 趋势 | 持续下降 | 不降（LR 不对）或震荡剧烈 |
| 最终 Loss | < 0.005 | > 0.02（epochs 不够或 LR 不对）|
| GPU 显存 | 16-22 GB | OOM（改 pooling=4）|
| 8 个 missing keys | ✅ 正常 | 其他数量（检查配置）|

---

## 输出文件

训练完成后，`checkpoints/phase1/` 目录下：

```
checkpoints/phase1/
├── phase1_epoch5.pt     # 第 5 轮保存
├── phase1_epoch10.pt
├── phase1_epoch15.pt
└── phase1_epoch20.pt    # 最终模型
```

每个 `.pt` 文件包含：

| Key | 内容 |
|-----|------|
| `model_state_dict` | 模型权重 |
| `optimizer_state_dict` | 优化器状态（可恢复训练）|
| `epoch` | 训练轮数 |
| `loss` | 该 epoch 的平均 loss |

---

## Step 4：推理 — 生成不同距离的太阳图

```bash
python surya_orbit/infer_phase1.py \
    --checkpoint checkpoints/phase1/phase1_epoch20.pt \
    --input-nc data/SDO_20141023/20141023_1200.nc \
    --distances 0.83 1.0 1.5 2.5 3.2 \
    --output-dir outputs/phase1_inference
```

输出：

```
outputs/phase1_inference/
├── input_1AU.png           # 输入图像预览
├── scaled_0.83AU.png       # 各距离的预测图像
├── scaled_1.00AU.png
├── scaled_1.50AU.png
├── scaled_2.50AU.png
├── scaled_3.20AU.png
├── comparison_labels.png   # 并排对比
├── sun_diameter.csv        # 太阳直径统计
└── prediction.nc           # NetCDF 格式预测
```

`sun_diameter.csv` 内容示例：

```
 distance_au  sun_diameter_px  expected_diameter_px  scale_factor
        0.83             1984                   1963          1.20
        1.00             1640                   1628          1.00
        1.50             1098                   1085          0.67
        2.50              652                    651          0.40
        3.20              510                    509          0.31
```

**判断标准**：`sun_diameter_px` 随距离增大而减小，且与 `expected_diameter_px` 接近。

---

## Step 5：验证 — 自动化测试

```bash
# 基础烟雾测试（不需要真实数据）
python surya_orbit/test_phase1.py \
    --checkpoint checkpoints/phase1/phase1_epoch20.pt

# 完整测试（需要真实 NC 文件）
python surya_orbit/test_phase1.py \
    --checkpoint checkpoints/phase1/phase1_epoch20.pt \
    --input-nc data/SDO_20141023/20141023_0600.nc
```

测试项：

| 编号 | 测试 | 验证什么 | 需要 NC 文件 |
|------|------|----------|-------------|
| 001 | 烟雾测试 | 模型能加载，forward 不报错，输出无 NaN | ❌ |
| 002 | 缩放方向 | 0.83 AU 太阳 > 3.20 AU 太阳 | ✅ |
| 003 | 缩放比 | 直径比 ≈ 距离反比（误差 < 15%） | ✅ |
| 004 | 恒等性 | 1 AU → 1 AU 输出 ≈ 输入 | ✅ |
| 005 | 推理速度 | 单次 < 2 秒 | ✅ |

通过输出示例：

```
==============================================================
Phase 1 Validation Tests
Device: cuda
Config: surya_orbit/config_phase1.yaml
Checkpoint: checkpoints/phase1/phase1_epoch20.pt
==============================================================

[TEST 001] Smoke test — model loads and runs forward ...
  ✅ PASS — output shape (1,13,2048,2048), no NaN/Inf

[TEST 002] Scale direction — farther → smaller sun ...
  Sun diameter at 0.83 AU: 1984 px
  Sun diameter at 3.20 AU: 510 px
  Ratio far/near = 0.257 (expected ~0.26)
  ✅ PASS — sun smaller at larger distance

[TEST 003] Scale ratio — diameter ∝ 1/distance ...
  Mean ratio error: 3.2%
  Max  ratio error: 7.1%
  ✅ PASS — diameter ratios within 15% of expected

[TEST 004] Identity — 1 AU → 1 AU should ≈ input ...
  Mean absolute difference: 0.023456
  ✅ PASS — identity prediction close to input

[TEST 005] Inference speed ...
  Average: 342 ms/batch (batch_size=1)
  ✅ PASS — inference within time limit

==============================================================
All tests passed. ✅
==============================================================
```

---

## 常见问题

### OOM（显存不足）

```
RuntimeError: CUDA out of memory
```

**解决**：修改 `config_phase1.yaml` 中 `data.pooling: 4`，以及 `data.img_size_after_pool: 1024`。

### 变量名不匹配

```
KeyError: "Variable 'aia094' not found in dataset"
```

**解决**：用 Step 1.5 的命令查 NC 文件实际变量名，然后修改 config 中 `sdo_channels`。

### 下载数据失败

```
FileNotFoundError: No .nc files found in data/SDO_20141023
```

**解决**：确认 AWS CLI 已安装（`aws --version`），确认网络能访问 `s3://nasa-surya-bench`。

---

## Phase 1 → Phase 2

Phase 1 完成后（几何缩放正确），进入 Phase 2 纹理训练：

1. 加载 Phase 1 的 `phase1_epoch20.pt`
2. 冻结 `OrbitAwareFlowModel`（几何能力保留）
3. 解冻各层 `adaLN_modulation` + 添加 LoRA
4. 新增物理约束损失：通量守恒、临边昏暗、结构保持
5. 继续训练 10-20 epochs
