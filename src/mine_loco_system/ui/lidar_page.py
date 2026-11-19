# -*- coding: utf-8 -*-
"""激光雷达模块：双程消光点云仿真与障碍物判决（E3 模型嵌入）。"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                               QPushButton, QGridLayout, QCheckBox,
                               QFrame)

from . import theme
from .widgets import Card, make_plot, export_csv, KpiCard
from ..core import lidar as L
from ..core.lidar import LidarSession, reliable_range


class LidarPage(QFrame):
    ID = "lidar"
    TITLE = "③ 激光雷达 · 障碍物检测"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.ctx = ctx
        self.session = LidarSession(beta=0.0, seed=23)
        self.log = []
        self.frame_log = []
        self._keep = []
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.setInterval(80)
        self._timer.start()
        self._alarmed = False

    # -------------------------------------------------------------- layout
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        root.addLayout(left, 2)
        c_ctl = Card("仿真控制")
        gl = QGridLayout()
        gl.addWidget(QLabel("粉尘消光 β"), 0, 0)
        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.setRange(0, 40)
        self.b_slider.setValue(0)
        self.b_slider.valueChanged.connect(self.on_beta)
        gl.addWidget(self.b_slider, 0, 1)
        self.b_lbl = QLabel("0.00 /m（V=200 m）")
        self.b_lbl.setStyleSheet("font-weight:700; color:%s;" % theme.WARN)
        gl.addWidget(self.b_lbl, 0, 2)
        self.obs_chk = QCheckBox("强制每帧存在障碍物（演示用）")
        self.obs_chk.setChecked(True)
        gl.addWidget(self.obs_chk, 1, 0, 1, 3)
        c_ctl.body().addLayout(gl)
        bl = QHBoxLayout()
        b2 = QPushButton("⏸ 暂停")
        b2.setObjectName("Primary")
        b2.clicked.connect(self.toggle)
        b3 = QPushButton("重置统计")
        b3.clicked.connect(self.reset_stats)
        bl.addWidget(b2)
        bl.addWidget(b3)
        c_ctl.body().addLayout(bl)
        b_exp = QPushButton("导出逐帧判决 CSV")
        b_exp.clicked.connect(self.export)
        c_ctl.body().addWidget(b_exp)
        left.addWidget(c_ctl)

        self.kpi_det = KpiCard("检出率（累计）", "--", "判据：候选簇≥2 点且误差<0.5 m")
        self.kpi_fp = KpiCard("虚警率（累计）", "--", "无障碍帧误报比例")
        self.kpi_rmse = KpiCard("测距 RMSE", "--", "障碍物距离估计")
        self.kpi_valid = KpiCard("点云有效率", "--", "有效回波 / 481 束")
        for k in (self.kpi_det, self.kpi_fp, self.kpi_rmse, self.kpi_valid):
            left.addWidget(k)
        left.addStretch(1)
        self.range_lbl = QLabel("--")
        self.range_lbl.setObjectName("Dim")
        self.range_lbl.setWordWrap(True)
        left.addWidget(self.range_lbl)

        # 右：极坐标点云 + 判决结果
        right = QVBoxLayout()
        root.addLayout(right, 8)
        self.scan_plot = pg.PlotWidget()
        self.scan_plot.setBackground("#0b0f14")
        self.scan_plot.setTitle("二维激光雷达扫描（俯视，车头位于原点）",
                                color=theme.FG, size="12pt")
        self.scan_plot.showGrid(x=True, y=True, alpha=0.2)
        self.scan_plot.setXRange(-12, 12)
        self.scan_plot.setYRange(-2.5, 26)
        self.scan_plot.setAspectLocked(True)
        right.addWidget(self.scan_plot, 5)
        self.scat_wall = pg.ScatterPlotItem(pen=None, size=4,
                                            brush=pg.mkBrush("#557090"))
        self.scat_obs = pg.ScatterPlotItem(pen=None, size=6,
                                           brush=pg.mkBrush(theme.OK))
        self.scat_noise = pg.ScatterPlotItem(pen=None, size=4,
                                             brush=pg.mkBrush(theme.ALARM))
        self.scan_plot.addItem(self.scat_wall)
        self.scan_plot.addItem(self.scat_obs)
        self.scan_plot.addItem(self.scat_noise)
        # 通行区边界线
        for yy in (0.90, -0.90):
            ln = pg.InfiniteLine(pos=yy, angle=0,
                                 pen=pg.mkPen(theme.ACCENT, style=Qt.DotLine))
            self.scan_plot.addItem(ln)
        # 判决条
        self.verdict_lbl = QLabel("等待数据…")
        self.verdict_lbl.setStyleSheet("font-size:16px; font-weight:700;")
        self.verdict_lbl.setAlignment(Qt.AlignCenter)
        right.addWidget(self.verdict_lbl)
        self.detail_lbl = QLabel("--")
        self.detail_lbl.setObjectName("Dim")
        self.detail_lbl.setAlignment(Qt.AlignCenter)
        right.addWidget(self.detail_lbl)
        self._keep += [left, right, gl, bl, c_ctl]

    # ------------------------------------------------------------ handlers
    def on_beta(self):
        b = self.b_slider.value() / 100.0
        self.session.beta = b
        V = 3.0 / max(b, 1e-6)
        self.b_lbl.setText("%.2f /m（V=%.0f m）" % (b, V))
        r = reliable_range(V)
        self.range_lbl.setText(
            "理论可靠检出距离 %.1f m（双程消光，I_thr=0.14）。"
            "β≥0.30 时检出率归零——该限制无法通过算法绕过，需多回波体制或视觉-雷达互补。" % r)

    def toggle(self):
        if self._timer.isActive():
            self._timer.stop()
            self.sender().setText("▶ 继续")
        else:
            self._timer.start()
            self.sender().setText("⏸ 暂停")

    def reset_stats(self):
        self.session.reset()
        self.frame_log = []

    def tick(self):
        info = self.session.step(
            force_obstacle=True if self.obs_chk.isChecked() else None)
        self.frame_log.append({
            "帧": self.session.frames,
            "beta": self.session.beta,
            "有障碍": int(info["has_obstacle"]),
            "障碍距离m": round(info["obs_d"], 2) if info["obs_d"] else "",
            "判决": info["verdict"],
            "估计距离m": round(info["estimate"], 2) if info["estimate"] is not None else "",
            "候选点数": info["n_cand"],
            "有效回波": info["n_echo"],
        })
        self._draw(info)
        m = self.session.metrics()
        self.kpi_det.set("%.1f %%" % m["检出率_%"],
                         color=theme.OK if m["检出率_%"] > 80 else theme.ALARM)
        self.kpi_fp.set("%.1f %%" % m["虚警率_%"])
        self.kpi_rmse.set(("%.1f cm" % (m["测距RMSE_m"] * 100))
                          if m["测距RMSE_m"] is not None else "--")
        self.kpi_valid.set("%.1f %%" % m["点云有效率_%"])
        # β≥0.3 报警（一次）
        if self.session.beta >= 0.3 and self.session.frames > 30 and not self._alarmed:
            self._alarmed = True
            self.ctx.alarms.emit_alarm(
                "激光雷达", "alarm",
                "β=%.2f /m 高粉尘：雷达点云有效率仅 %.1f%%，检出率 %.1f%%——"
                "回波被双程消光压制"
                % (self.session.beta, m["点云有效率_%"], m["检出率_%"]),
                advice="建议切换视觉-雷达互补模式：V=20 m 时视觉在 6 m 处仍保留 40.7% 对比度，"
                       "可覆盖雷达盲区。")

    def _draw(self, info):
        d, rho = info["d_meas"], info["rho_meas"]
        ok = np.isfinite(d)
        x = d * L.SIN_A
        y = d * np.cos(L.SIN_A)
        wall = ok & (rho < 0.10) & ok
        obs = ok & (rho >= 0.10) & (d >= L.D_MIN) & (d < L.DET_RANGE)
        noise = ok & (d < L.D_MIN)
        self.scat_wall.setData(x[wall], y[wall])
        self.scat_obs.setData(x[obs], y[obs])
        self.scat_noise.setData(x[noise], y[noise])
        v = info["verdict"]
        if v == "hit":
            self.verdict_lbl.setText(
                "● 检出障碍物 ｜ 距离 %.2f m（真值 %.2f m）"
                % (info["estimate"], info["obs_d"]))
            self.verdict_lbl.setStyleSheet(
                "font-size:16px; font-weight:700; color:%s;" % theme.ALARM)
            self.detail_lbl.setText("候选回波 %d 点 ｜ 通行区 |y|≤0.9 m ｜ 强度≥0.12"
                                    % info["n_cand"])
        elif v == "miss":
            self.verdict_lbl.setText("● 有障碍但未检出（回波不足或丢失）")
            self.verdict_lbl.setStyleSheet(
                "font-size:16px; font-weight:700; color:%s;" % theme.WARN)
            self.detail_lbl.setText("候选回波 %d 点（<2 点不上报）" % info["n_cand"])
        elif v == "clear":
            self.verdict_lbl.setText("● 通行区畅通")
            self.verdict_lbl.setStyleSheet(
                "font-size:16px; font-weight:700; color:%s;" % theme.OK)
            self.detail_lbl.setText("有效回波 %d / 481" % info["n_echo"])
        else:
            self.verdict_lbl.setText("● 虚警（无障碍误报）")
            self.verdict_lbl.setStyleSheet(
                "font-size:16px; font-weight:700; color:%s;" % theme.WARN)
            self.detail_lbl.setText("候选回波 %d 点" % info["n_cand"])

    def export(self):
        rows = [[r["帧"], r["beta"], r["有障碍"], r["障碍距离m"], r["判决"],
                 r["估计距离m"], r["候选点数"], r["有效回波"]]
                for r in self.frame_log]
        export_csv(self, "雷达逐帧判决.csv",
                   ["帧", "beta", "有障碍", "障碍距离m", "判决", "估计距离m",
                    "候选点数", "有效回波"], rows)
