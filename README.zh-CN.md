# Mine-Loco Perception

**基于物理引导域适应的井下电机车智能感知与运维系统**

[English](README.md) · [文档](docs/) · [项目主页](https://jinnn.xyz/mine-loco/)

---

## 这是什么

井下电机车面临“没有标注数据”的困境：粉尘与低照度让视觉退化，而井下轨道障碍物数据难以大规模采集。
本项目用**大气散射模型**合成粉尘退化，以矿山可直接测量的**能见度 V** 为控制变量（`β = 3/V`），
并给出三件事：

1. **一套可复现的评测协议**：训练 / 标定 / 评测三刀切开，退化种子互异；
2. **一次如实报告的失败**：原实现中“标定集与评测集同源”，修正后原 **+3.89 pp** 增益不成立（8 次对照 0/8）；
3. **一次规模扩大后的反转 + 消融**：5000 张 / 640 px / 80 轮下，物理引导模型在粉尘工况
   **+10.15 / +10.93 pp**（V=20 m），而全局灰度对齐的噪声增广、仅光度增广均无增益——
   增益来自**随深度变化的散射结构**本身。

> **边界**：粉尘为合成退化；ExDark 类别与井下轨道障碍物不一致；扩训为单次运行。
> 本项目验证的是方法有效性与评测协议，不是井下实检精度。

---

## 主要结果

`mAP@0.5 (%)`，368 张独立评测集，5000 张 / 640 px / 80 轮：

| 评测集 | M_clean | M_phys | M_noise | M_hsv | Δ phys−clean |
|---|---:|---:|---:|---:|---:|
| 干净 | 61.81 | 42.61 | 42.70 | 55.46 | −19.20 |
| V=30 m | 23.04 | **32.96** | 20.35 | 22.67 | +9.92 |
| V=20 m | 22.09 | **32.24** | 18.76 | 21.65 | +10.15 |
| V=15 m | 21.06 | **30.61** | 19.25 | 19.86 | +9.55 |
| V=10 m | 17.96 | **26.85** | 18.20 | 16.35 | +8.89 |

部署侧：

| 模型 / 推理方式 | 干净 | V=20 m | V=10 m |
|---|---:|---:|---:|
| M_clean | 61.81 | 22.09 | 17.96 |
| M_clean + TTA | 61.05 | 21.70 | 17.62 |
| M_phys | 42.60 | 32.24 | 26.85 |
| **M_phys + TTA** | **44.88** | **34.17** | **27.60** |
| 模型汤（权重平均） | 51.01 | 29.09 | 22.72 |

- TTA 对 M_phys 三个域全正；模型汤把干净域代价从 −19.2 pp 收窄到 −10.8 pp。
- `M_phys` 640 px 单张推理约 **9.2 ms（109 FPS）**，TTA 约 **20.8 ms（48 FPS）**（RTX 5060 Laptop）。

完整表格、小规模负结果与逐类行为见 [docs/results.md](docs/results.md)；
方法见 [docs/method.md](docs/method.md)。

---

## 仓库结构

```
src/mine_loco_system/   PySide6 六页面演示系统（物理/融合/雷达/健康/回放）
scripts/                训练、评测、复现、基准脚本
tests/                  18 项 pytest（物理/融合/雷达/健康）
docs/                   方法、结果、复现、部署、FAQ
assets/                 图件与小型参考 JSON
site/                   静态项目主页（可部署到 jinnn.xyz）
data/                   数据集（不提交，见 data/README.md）
weights/                权重（不提交，见 weights/README.md）
outputs/                生成的案例、训练产物、结果、图件（git 忽略）
```

---

## 快速开始

```bash
git clone https://github.com/<your-org>/mine-loco-perception.git
cd mine-loco-perception
pip install -e ".[full]"
pytest -q

python scripts/download_exdark.py        # ExDark (CC BY 4.0) → data/exdark/data
python -m mine_loco_system.main          # 启动演示系统
```

权重：将 `yolov8n.pt` 放入 `weights/`；训练权重不随仓库提交，可用
`scripts/exp_scaleup_ablation.py` 复现四个模型。

### 复现命令

```bash
python scripts/rerun_e2_offset.py
python scripts/rerun_e3_lidar.py
python scripts/rerun_e4_lifetime.py
python scripts/sensitivity_check.py
python scripts/redraw_figs.py
python scripts/exp_real_detect_fixed.py --epochs 15 --batch 16
python scripts/exp_scaleup_ablation.py --epochs 80 --batch 16 --imgsz 640
python scripts/eval_extension.py
```

`scripts/check_repro_requirements.py` 会逐项打印数据、权重、GPU、网络的 PASS/FAIL。

---

## 网页部署

`site/` 是零依赖静态页，本地预览：

```bash
cd site && python -m http.server 8000
```

部署到 **jinnn.xyz** 的部署方式（GitHub Pages 子路径/子域、Cloudflare Pages、VPS + Nginx）见
[docs/deployment.md](docs/deployment.md)。仓库已包含 GitHub Pages 工作流
`.github/workflows/pages.yml`。

---

## 引用

```bibtex
@software{mine_loco_perception_2026,
  title  = {Mine-Loco Perception: Physics-Guided Domain Adaptation for
            Underground Mine Locomotive Perception and Maintenance},
  author = {Li, Kerui and Wang, Zhiwen and Yang, Moran and Jin, Zikuan and
            Qi, Gaoyang and Xia, Yuhan and Liu, Yixuan},
  year   = {2026},
  url    = {https://github.com/<your-org>/mine-loco-perception}
}
```

## 许可与致谢

- 代码：**MIT**（见 [LICENSE](LICENSE)）。
- ExDark 数据集：**CC BY 4.0**，版权归其原作者；使用请引用 ExDark。
- 粉尘为合成退化，不含真实井下图像。
- 本仓库只包含代码与小型图件；申报书、个人材料、未公开专利草稿不在其中。
