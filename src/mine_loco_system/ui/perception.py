# -*- coding: utf-8 -*-
"""视觉感知模块：物理退化 → 检测，演示域差异与能见度参数化。"""
import os
import random

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                               QCheckBox, QPushButton, QComboBox,
                               QDoubleSpinBox, QFileDialog, QSizePolicy,
                               QFrame)

from . import theme
from .widgets import Card, make_plot, export_csv
from ..core.physics import (degrade_pil_image, degrade_array, depth_map,
                            contrast_retention, beta_from_visibility,
                            mean_transmission, degrade_uniform_noise,
                            photometric_jitter)
from ..core.detector import Detector, WEIGHT_CHOICES

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

from ..paths import default_image_dir

DEFAULT_DIR = default_image_dir()

MODES = ["物理退化（大气散射）", "均匀噪声（非物理对照）", "光度抖动（仅HSV类）",
         "原始（无退化）"]

CLS_COLOR = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
             "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324"]


def np_to_pixmap(arr):
    h, w, ch = arr.shape
    img = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


class PerceptionPage(QFrame):
    ID = "perception"
    TITLE = "① 视觉感知 · 障碍物检测"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.ctx = ctx
        self.det = Detector()
        self.det_imgsz = 416
        self.files = []
        self.file_idx = 0
        self.orig_arr = None
        self.depth = None
        self.degraded_arr = None
        self.enhanced_arr = None
        self.dets = []
        self.playing = False
        self.last_dt = 0.0
        self.enhance_on = False          # 与报告口径一致：默认关闭
        self._keep = []
        self._build()
        self.ctx.state.changed.connect(self.on_condition)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.on_next)
        self._timer.setInterval(1200)
        self.load_dir(DEFAULT_DIR, silent=True)

    # -------------------------------------------------------------- layout
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # 左：控制
        left = QVBoxLayout()
        root.addLayout(left, 2)

        c_data = Card("工况数据加载")
        bl = QVBoxLayout()
        self.dir_lbl = QLabel("数据源：--")
        self.dir_lbl.setObjectName("Dim")
        self.dir_lbl.setWordWrap(True)
        bl.addWidget(self.dir_lbl)
        b1 = QPushButton("加载图像目录…")
        b1.clicked.connect(self.pick_dir)
        b2 = QPushButton("上一张")
        b2.clicked.connect(lambda: self.step(-1))
        b3 = QPushButton("下一张")
        b3.clicked.connect(lambda: self.step(1))
        self.play_btn = QPushButton("▶ 连续播放")
        self.play_btn.setObjectName("Primary")
        self.play_btn.clicked.connect(self.toggle_play)
        hb = QHBoxLayout()
        hb.addWidget(b2)
        hb.addWidget(b3)
        hb.addWidget(self.play_btn)
        bl.addLayout(hb)
        bl.addWidget(b1)
        c_data.body().addLayout(bl)
        left.addWidget(c_data)

        c_deg = Card("粉尘工况（能见度参数化）")
        dl = QVBoxLayout()
        self.mode_box = QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.currentTextChanged.connect(lambda _: self.refresh())
        dl.addWidget(self.mode_box)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("能见度 V"))
        self.v_slider = QSlider(Qt.Horizontal)
        self.v_slider.setRange(5, 200)
        self.v_slider.setValue(20)
        self.v_slider.valueChanged.connect(self.on_condition)
        hl.addWidget(self.v_slider, 1)
        self.v_lbl = QLabel("20 m")
        self.v_lbl.setStyleSheet("font-weight:700; color:%s;" % theme.WARN)
        self.v_lbl.setMinimumWidth(70)
        hl.addWidget(self.v_lbl)
        dl.addLayout(hl)
        self.beta_lbl = QLabel("β = 0.150 /m ｜ 6 m 处对比度保留 40.7%")
        self.beta_lbl.setObjectName("Dim")
        dl.addWidget(self.beta_lbl)
        self.enh_chk = QCheckBox("启用前端图像增强（CLAHE，默认关闭——对照实验未取得正收益）")
        self.enh_chk.setChecked(False)
        self.enh_chk.toggled.connect(self.on_enhance)
        dl.addWidget(self.enh_chk)
        c_deg.body().addLayout(dl)
        left.addWidget(c_deg)

        c_det = Card("检测配置")
        dl2 = QVBoxLayout()
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("置信度阈值"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.valueChanged.connect(lambda _: self.refresh())
        hl2.addWidget(self.conf_spin)
        self.w_lbl = QLabel("权重（输入 %d px）" % self.det_imgsz)
        self.w_lbl.setObjectName("Dim")
        hl2.addWidget(self.w_lbl, 1)
        dl2.addLayout(hl2)
        # 模型选择：基线 / 扩训基线 / 物理引导（扩训消融实验模型）
        self.w_box = QComboBox()
        self._w_choices = []
        for name, path, isz in WEIGHT_CHOICES:
            if os.path.exists(path):
                self.w_box.addItem(name)
                self._w_choices.append((name, path, isz))
        cur = next((i for i, (n, p, s) in enumerate(self._w_choices)
                    if os.path.normpath(p) == os.path.normpath(
                        self.det.weights_path or "")), 0)
        if self._w_choices:
            self.w_box.setCurrentIndex(cur)
            self.det_imgsz = self._w_choices[cur][2]
        self.w_box.currentIndexChanged.connect(self.on_weight_change)
        dl2.addWidget(self.w_box)
        self.tta_chk = QCheckBox("TTA 增强推理（较慢，粉尘工况约 +1~2 pp）")
        self.tta_chk.setChecked(False)
        self.tta_chk.toggled.connect(lambda _: self.refresh())
        dl2.addWidget(self.tta_chk)
        self.det_status = QLabel("--")
        self.det_status.setObjectName("Dim")
        self.det_status.setWordWrap(True)
        dl2.addWidget(self.det_status)
        b_export = QPushButton("导出检测数据 CSV")
        b_export.clicked.connect(self.export)
        dl2.addWidget(b_export)
        c_det.body().addLayout(dl2)
        left.addWidget(c_det)
        left.addStretch(1)

        # 右：三联图 + 对比度曲线
        right = QVBoxLayout()
        root.addLayout(right, 8)
        imgs = QHBoxLayout()
        imgs.setSpacing(12)
        self.img_labels = []
        for cap in ("① 原始低照度", "② 粉尘退化工况", "③ 检测结果"):
            v = QVBoxLayout()
            lab = QLabel()
            lab.setMinimumSize(430, 300)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet(
                "background:#0b0f14; border:1px solid %s; border-radius:6px;"
                % theme.BORDER)
            cap_lbl = QLabel(cap)
            cap_lbl.setAlignment(Qt.AlignCenter)
            cap_lbl.setObjectName("Dim")
            v.addWidget(lab, 1)
            v.addWidget(cap_lbl)
            v.setAlignment(Qt.AlignHCenter)
            w = QFrame()
            w.setLayout(v)
            imgs.addWidget(w, 1)
            self.img_labels.append(lab)
        right.addLayout(imgs, 5)

        bottom = QHBoxLayout()
        self.cnt_lbl = QLabel("检测：-- ｜ 推理 -- ms")
        self.cnt_lbl.setObjectName("Dim")
        bottom.addWidget(self.cnt_lbl)
        bottom.addStretch(1)
        right.addLayout(bottom)

        self.ret_plot = make_plot("目标-背景对比度保留率 exp(−β·d) 随距离变化")
        self.ret_plot.addLegend()
        self.ret_plot.setXRange(0, 30)
        self.ret_plot.setYRange(0, 1.05)
        right.addWidget(self.ret_plot, 2)
        self._keep += [left, right, imgs, bl, hb, dl, dl2, hl, hl2, bottom, c_data, c_deg, c_det]

    # ------------------------------------------------------------ handlers
    def load_dir(self, d, silent=False):
        if not os.path.isdir(d):
            if not silent:
                self.ctx.toast("目录不存在：%s" % d)
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.files = sorted(f for f in os.listdir(d) if f.lower().endswith(exts))
        self.dir_lbl.setText("数据源：%s（%d 张）" % (d, len(self.files)))
        self._dir = d
        self.file_idx = 0
        if self.files:
            self.load_image(os.path.join(d, self.files[0]))
            self._timer.start() if self.playing else None

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图像目录", DEFAULT_DIR)
        if d:
            self.load_dir(d)

    def load_image(self, path):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w = 640
                nh = max(2, int(w * im.height / im.width))
                im2 = im.resize((w, nh), LANCZOS)
                self.orig_arr = np.asarray(im2).copy()
        except Exception as e:
            self.ctx.toast("图像解码失败：%s" % e)
            return
        self.depth = depth_map(self.orig_arr.shape[0], self.orig_arr.shape[1])
        self.path = path
        self.refresh()

    def on_condition(self):
        V = self.v_slider.value()
        self.v_lbl.setText(f"{V} m")
        b = beta_from_visibility(V)
        c6 = contrast_retention(b, 6.0)
        self.beta_lbl.setText(
            f"β = {b:.3f} /m ｜ 6 m 处对比度保留 {c6*100:.1f}%")
        self.ctx.state.set_V(V)
        self.refresh()

    def on_weight_change(self, idx):
        """切换检测权重（基线 / 扩训基线 / 物理引导），输入尺寸随之切换。"""
        if not self._w_choices:
            return
        name, path, isz = self._w_choices[max(0, min(idx, len(self._w_choices) - 1))]
        self.det = Detector(weights=path)
        self.det_imgsz = isz
        self.w_lbl.setText("权重（输入 %d px）" % isz)
        self.ctx.alarms.emit_alarm(
            "视觉感知", "info",
            "检测模型切换：%s（扩训消融实验模型，独立评测协议见"
            "《扩训与消融实验说明.md》）" % name)
        self.refresh()

    def on_enhance(self):
        self.enhance_on = self.enh_chk.isChecked()
        self.ctx.alarms.emit_alarm(
            "视觉感知", "info",
            "前端图像增强%s" % ("已启用（受约束可选模块）" if self.enhance_on
                               else "已关闭（默认，对照实验未取得正收益）"))
        self.refresh()

    def _current_degraded(self):
        V = self.v_slider.value()
        mode = self.mode_box.currentText()
        seed = self.file_idx
        if mode.startswith("物理"):
            return degrade_array(self.orig_arr, self.depth, V, seed)
        if mode.startswith("均匀"):
            t = mean_transmission(self.depth, V)
            return degrade_uniform_noise(self.orig_arr, t, seed)
        if mode.startswith("光度"):
            return photometric_jitter(self.orig_arr, seed)
        return self.orig_arr.copy()

    def refresh(self):
        if self.orig_arr is None:
            return
        self.degraded_arr = self._current_degraded()
        if self.enhance_on:
            self.degraded_arr = self._apply_clahe(self.degraded_arr)
        # 显示前两联
        self.img_labels[0].setPixmap(self._fit(self.orig_arr))
        self.img_labels[1].setPixmap(self._fit(self.degraded_arr))
        # 检测
        dets, dt, err = self.det.infer(self.degraded_arr,
                                       conf=self.conf_spin.value(),
                                       imgsz=self.det_imgsz,
                                       augment=self.tta_chk.isChecked())
        self.dets = dets
        self.last_dt = dt
        self._draw_dets()
        n = len(dets)
        names = "、".join(sorted({d["name"] for d in dets})) or "无"
        if err:
            self.det_status.setText("检测器未就绪：%s" % err)
        else:
            self.det_status.setText(
                "检测器就绪（YOLOv8n，%d 类）｜ 当前检出 %d 目标：%s"
                % (len(self.det.names), n, names))
        self.cnt_lbl.setText(
            "检测：%d 目标 ｜ 推理 %.0f ms ｜ V=%d m ｜ 文件 %s"
            % (n, self.last_dt * 1000, self.v_slider.value(),
               os.path.basename(getattr(self, "path", ""))))
        self._plot_retention()

    def _apply_clahe(self, arr):
        """CLAHE 增强（报告 4.5 节的受约束可选模块）。用 cv2 实现。"""
        try:
            import cv2
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception:
            return arr

    def _draw_dets(self):
        pm = self._fit(self.degraded_arr)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont("Microsoft YaHei", 10, QFont.Bold)
        p.setFont(f)
        scale = pm.width() / self.degraded_arr.shape[1]
        for d in self.dets:
            x1, y1, x2, y2 = [v * scale for v in d["bbox"]]
            color = QColor(CLS_COLOR[d["cls"] % len(CLS_COLOR)])
            pen = QPen(color, 2)
            p.setPen(pen)
            p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            p.fillRect(int(x1), int(y1) - 18, 88, 17, color)
            p.setPen(QPen(QColor("#101418")))
            p.drawText(int(x1) + 2, int(y1) - 5,
                       "%s %.0f%%" % (d["name"], d["conf"] * 100))
        p.end()
        self.img_labels[2].setPixmap(pm)

    def _fit(self, arr):
        pm = np_to_pixmap(arr)
        return pm.scaled(self.img_labels[1].size(), Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)

        self._keep += [left, right, imgs, bl, hb, dl, dl2, hl, hl2, bottom, c_data, c_deg, c_det]

    def _plot_retention(self):
        import numpy as _np
        d = _np.linspace(1, 30, 100)
        b = beta_from_visibility(self.v_slider.value())
        y = _np.exp(-b * d)
        self.ret_plot.clear()
        pen = pg.mkPen(theme.WARN, width=2)
        self.ret_plot.plot(d, y, pen=pen, name="V=%d m" % self.v_slider.value())
        # 参考曲线
        for V, c in ((200, "#557"), (50, "#7a8ba0"), (10, theme.ALARM)):
            y2 = _np.exp(-(3.0 / V) * d)
            self.ret_plot.plot(d, y2, pen=pg.mkPen(c, width=1, style=Qt.DashLine),
                               name="V=%d m" % V)
        # 目标距离标记（制动距离量级）
        line = pg.InfiniteLine(pos=6.0, angle=90, pen=pg.mkPen(theme.ACCENT,
                                                               style=Qt.DotLine))
        self.ret_plot.addItem(line)
        self.ret_plot.addLegend()

    def step(self, d):
        if not self.files:
            return
        self.file_idx = (self.file_idx + d) % len(self.files)
        self.load_image(os.path.join(self._dir, self.files[self.file_idx]))

    def on_next(self):
        self.step(1)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("⏸ 暂停" if self.playing else "▶ 连续播放")
        if self.playing:
            self._timer.start()
        else:
            self._timer.stop()

    def export(self):
        rows = [[d["name"], *[round(v, 1) for v in d["bbox"]],
                 round(d["conf"], 3)] for d in self.dets]
        export_csv(self, "检测结果.csv",
                   ["类别", "x1", "y1", "x2", "y2", "置信度"], rows)

    def demo_step(self, i):
        """演示模式脚本步骤。"""
        scenes = [(200, 0), (50, 1), (20, 2), (10, 3)]
        V, fi = scenes[min(i, len(scenes) - 1)]
        self.v_slider.setValue(V)
        self.file_idx = fi % max(len(self.files), 1)
        if self.files:
            self.load_image(os.path.join(self._dir, self.files[self.file_idx]))
