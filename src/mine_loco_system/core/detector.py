# -*- coding: utf-8 -*-
"""YOLOv8n 检测器封装：加载项目训练权重，对退化图像做推理。

默认加载 ``weights/m_clean_800/best.pt``（干净训练基线），若不存在则回退到
扩训 M_clean 或 ``weights/yolov8n.pt``。基线模型用于演示其在高粉尘工况下的
性能衰减；界面可切换到 M_phys（物理引导）模型做同图对比。
"""
import os

from ..paths import default_weight, scaleup_weight, WEIGHTS_DIR

NAMES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
         "Cup", "Dog", "Motorbike", "People", "Table"]

# 可选模型清单（名称, 权重路径, 匹配推理输入尺寸）。
# M_phys 为 5000 张 / 640 px / 80 轮的物理引导模型；
# 其在粉尘工况的增益以独立评测协议为准（见 docs/results.md）。
WEIGHT_CHOICES = []
_w800 = WEIGHTS_DIR / "m_clean_800" / "best.pt"
if _w800.exists():
    WEIGHT_CHOICES.append(("M_clean 基线（800张/416）", str(_w800), 416))
_wc = scaleup_weight("M_clean")
_wp = scaleup_weight("M_phys")
if _wc.exists():
    WEIGHT_CHOICES.append(("M_clean 扩训基线（5000张/640）", str(_wc), 640))
if _wp.exists():
    WEIGHT_CHOICES.append(("M_phys 物理引导（5000张/640）", str(_wp), 640))


def find_weights():
    return default_weight()


class Detector:
    """轻量封装：延迟加载、CPU/GPU 自适应、返回统一格式结果。"""

    def __init__(self, weights=None, device=None):
        self.model = None
        self.names = NAMES
        self.device = device or os.environ.get('MINE_LOCO_DEVICE') or None
        self.weights_path = weights or find_weights()
        self.load_error = None
        self._try_load()

    def _try_load(self):
        if not self.weights_path:
            self.load_error = "未找到权重文件"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            try:
                self.names = [self.model.names[i] for i in range(len(
                    self.model.names))]
            except Exception:
                pass
        except Exception as e:                           # pragma: no cover
            self.model = None
            self.load_error = str(e)

    @property
    def ready(self):
        return self.model is not None

    def infer(self, arr_or_path, conf=0.25, imgsz=416, augment=False):
        """推理。输入 uint8 HWC 数组或路径，返回检测列表与耗时。

        augment=True 时启用 TTA（多尺度/翻转增强），粉尘工况通常 +1~2 pp，
        推理耗时约 2 倍。
        """
        if not self.ready:
            return [], 0.0, self.load_error
        import time
        t0 = time.time()
        kw = dict(conf=conf, imgsz=imgsz, verbose=False,
                  device=self.device or None, augment=bool(augment))
        r = self.model.predict(arr_or_path, **kw)[0]
        dt = time.time() - t0
        dets = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
            for i in range(len(xyxy)):
                cls_id = int(boxes.cls[i].item())
                dets.append({
                    "bbox": [float(v) for v in xyxy[i]],
                    "conf": float(boxes.conf[i].item()),
                    "cls": cls_id,
                    "name": (self.names[cls_id]
                             if 0 <= cls_id < len(self.names) else str(cls_id)),
                })
        return dets, dt, None
