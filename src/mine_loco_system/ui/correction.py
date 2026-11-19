# -*- coding: utf-8 -*-
"""融合纠偏模块：视觉-惯性抗差融合实时演示（E2 模型嵌入）。"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                               QPushButton, QCheckBox, QComboBox, QGridLayout,
                               QFrame)

from . import theme
from .widgets import Card, make_plot, export_csv, KpiCard
from ..core.fusion import FusionSession, run_e2_full, DT


class CorrectionPage(QFrame):
    ID = "correction"
    TITLE = "② 融合纠偏 · 抗差状态估计"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.ctx = ctx
        self.session = None
        self.log = []                      # 逐帧记录（导出用）
        self.odom = 0.0
        self._keep = []
        self._build()
        self.new_session()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.setInterval(33)        # 30 Hz
        self._timer.start()
        self.ctx.state.changed.connect(self.on_condition)

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
        self.b_slider.setValue(30)
        gl.addWidget(self.b_slider, 0, 1)
        self.b_lbl = QLabel("0.30 /m")
        self.b_lbl.setStyleSheet("font-weight:700; color:%s;" % theme.WARN)
        gl.addWidget(self.b_lbl, 0, 2)
        gl.addWidget(QLabel("轨迹类型"), 1, 0)
        self.kind_box = QComboBox()
        self.kind_box.addItems(["irregular（不规则摆动）", "straight（直线）",
                                "curve（弯道）"])
        gl.addWidget(self.kind_box, 1, 1, 1, 2)
        self.prior_chk = QCheckBox("轨道几何约束（量值域 ±0.15 m）")
        self.prior_chk.setChecked(True)
        gl.addWidget(self.prior_chk, 2, 0, 1, 3)
        self.gate_chk = QCheckBox("3σ 新息门限 + 值域门限（抗差机制）")
        self.gate_chk.setChecked(True)
        gl.addWidget(self.gate_chk, 3, 0, 1, 3)
        c_ctl.body().addLayout(gl)
        bl = QHBoxLayout()
        b1 = QPushButton("重新开始")
        b1.clicked.connect(self.new_session)
        b2 = QPushButton("⏸ 暂停")
        b2.setObjectName("Primary")
        b2.clicked.connect(self.toggle)
        self.speed_btn = QPushButton("×4 加速")
        self.speed_btn.clicked.connect(self.toggle_speed)
        bl.addWidget(b1)
        bl.addWidget(b2)
        bl.addWidget(self.speed_btn)
        c_ctl.body().addLayout(bl)
        b_exp = QPushButton("导出逐帧数据 CSV")
        b_exp.clicked.connect(self.export)
        c_ctl.body().addWidget(b_exp)
        left.addWidget(c_ctl)

        self.kpi_cards = []
        for name in ("RMSE（本方案）", "RMSE（纯视觉）", "逐帧纠偏成功率",
                     "99分位误差"):
            k = KpiCard(name, "--", "")
            self.kpi_cards.append(k)
            left.addWidget(k)
        left.addStretch(1)

        self.reject_lbl = QLabel("抗差门限：剔除 0 帧（0.0%）")
        self.reject_lbl.setObjectName("Dim")
        self.reject_lbl.setWordWrap(True)
        left.addWidget(self.reject_lbl)

        # 右：曲线
        right = QVBoxLayout()
        root.addLayout(right, 8)
        self.track_plot = make_plot("横向偏移估计（30 Hz 实时）")
        self.track_plot.addLegend()
        self.track_plot.setYRange(-0.25, 0.25)
        self.track_plot.addLine(y=0.15, pen=pg.mkPen(theme.ALARM, style=Qt.DashLine,
                                                     width=1))
        self.track_plot.addLine(y=-0.15, pen=pg.mkPen(theme.ALARM, style=Qt.DashLine,
                                                      width=1))
        right.addWidget(self.track_plot, 5)
        self.c_truth = self.track_plot.plot(pen=pg.mkPen("#e8eef5", width=2),
                                            name="真值")
        self.c_vis = self.track_plot.plot(pen=pg.mkPen(theme.WARN, width=1),
                                          name="纯视觉（零阶保持）")
        self.c_est = self.track_plot.plot(pen=pg.mkPen(theme.ACCENT, width=2),
                                          name="融合+轨道几何约束")
        self.scat_rej = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(theme.ALARM), size=7)
        self.track_plot.addItem(self.scat_rej)

        self.err_plot = make_plot("估计误差 |e|（m）")
        self.err_plot.addLegend()
        self.err_plot.setYRange(0, 0.35)
        right.addWidget(self.err_plot, 3)
        self._keep += [left, right, gl, bl, c_ctl, self.kpi_cards[:]]
        self.ce_vis = self.err_plot.plot(pen=pg.mkPen(theme.WARN, width=1),
                                         name="纯视觉")
        self.ce_est = self.err_plot.plot(pen=pg.mkPen(theme.OK, width=2),
                                         name="本方案")
        line20 = pg.InfiniteLine(pos=0.020, angle=0,
                                 pen=pg.mkPen(theme.ACCENT, style=Qt.DotLine))
        self.err_plot.addItem(line20)

    # ------------------------------------------------------------ handlers
    def on_condition(self):
        b = self.ctx.state.beta
        self.b_slider.setValue(int(b * 100))
        self._update_b_lbl()

    def _update_b_lbl(self):
        b = self.b_slider.value() / 100.0
        self.b_lbl.setText("%.2f /m" % b)

    def new_session(self):
        b = self.b_slider.value() / 100.0
        kind = self.kind_box.currentText().split("（")[0]
        self.session = FusionSession(
            beta=b, n=300, seed=11 + int(b * 100) % 7,
            use_track_prior=self.prior_chk.isChecked(),
            use_fusion=True,
            use_gate=self.gate_chk.isChecked(),
            kind=self.kind_box.currentText().split("（")[0])
        self.log = []
        self.odom = 0.0
        for c in (self.c_truth, self.c_vis, self.c_est):
            c.setData([], [])
        self.scat_rej.setData([], [])
        self.ce_vis.setData([], [])
        self.ce_est.setData([], [])

    def toggle(self):
        if self._timer.isActive():
            self._timer.stop()
            self.sender().setText("▶ 继续")
        else:
            self._timer.start()
            self.sender().setText("⏸ 暂停")

    def toggle_speed(self):
        if self._timer.interval() == 33:
            self._timer.setInterval(8)
            self.speed_btn.setText("×1 速度")
        else:
            self._timer.setInterval(33)
            self.speed_btn.setText("×4 加速")

    def tick(self):
        s = self.session
        if s is None:
            return
        for _ in range(4 if self._timer.interval() == 8 else 1):
            info = s.step()
            if info is None:
                self._finish()
                return
            self.log.append(info)
            self.odom += self.ctx.state.speed * DT
        self._redraw()

    def _finish(self):
        summ = self.session.summary()
        if summ and self.log:
            self.ctx.alarms.emit_alarm(
                "融合纠偏", "info",
                "单段仿真完成：RMSE %.1f cm（纯视觉 %.1f cm，下降 %.1f%%），"
                "逐帧成功率 %.1f%%"
                % (summ["RMSE_本方案_m"] * 100, summ["RMSE_纯视觉_m"] * 100,
                   summ["RMSE下降_%"], summ["成功率_本方案_%"]))
        self.new_session()

    def _redraw(self):
        s = self.session
        n = s.i
        if n == 0:
            return
        t = np.arange(n) * DT
        self.c_truth.setData(t, s.truth[:n])
        self.c_vis.setData(t, s.z_hold[:n])
        self.c_est.setData(t, s.est[:n])
        rej_idx = [i for i in range(n) if not s.accepted[i]]
        if rej_idx:
            self.scat_rej.setData(
                [t[i] for i in rej_idx],
                [s.z_med[i] for i in rej_idx])
        self.ce_vis.setData(t, np.abs(s.z_hold[:n] - s.truth[:n]))
        self.ce_est.setData(t, np.abs(s.est[:n] - s.truth[:n]))
        summ = self.session.summary()
        self.kpi_cards[0].set("%.1f cm" % (summ["RMSE_本方案_m"] * 100),
                              "目标 ≤2.0 cm")
        self.kpi_cards[1].set("%.1f cm" % (summ["RMSE_纯视觉_m"] * 100),
                              "下降 %.0f%%" % max(0, summ["RMSE下降_%"]))
        self.kpi_cards[2].set("%.1f %%" % summ["成功率_本方案_%"],
                              "纯视觉 %.1f %%（|e|≤20mm）" % summ["成功率_纯视觉_%"])
        self.kpi_cards[3].set("%.1f cm" % (summ["P99_本方案_m"] * 100),
                              "纯视觉 %.1f cm" % (summ["P99_纯视觉_m"] * 100))
        self.reject_lbl.setText(
            "抗差门限：剔除 %d 帧（%.1f%%）——退化帧/丢失帧由惯性递推桥接"
            % (s.n_reject, summ["门限剔除帧比例_%"]))
        # 高粉尘报警
        if (s.beta >= 0.3 and n > 60 and summ["成功率_纯视觉_%"] < 60
                and not getattr(self, "_alarmed", False)):
            self._alarmed = True
            self.ctx.alarms.emit_alarm(
                "融合纠偏", "warn",
                "β=%.2f /m 高粉尘工况：纯视觉逐帧成功率已降至 %.1f%%，"
                "融合估计保持 %.1f%%"
                % (s.beta, summ["成功率_纯视觉_%"], summ["成功率_本方案_%"]),
                advice="建议保持融合模式运行；视觉观测置信度不足时自动由惯性递推桥接。")

    # 对外接口（总览页读取）
    def live_summary(self):
        if self.session and self.session.i > 30:
            return self.session.summary()
        return {}

    def error_series(self):
        s = self.session
        if s and s.i > 2:
            n = s.i
            t = (np.arange(n) * DT).tolist()
            return {"t": t,
                    "vision": np.abs(s.z_hold[:n] - s.truth[:n]).tolist(),
                    "est": np.abs(s.est[:n] - s.truth[:n]).tolist()}
        return None

    def odometer(self):
        return self.odom

    def last_offset(self):
        return float(self.session.est[max(self.session.i - 1, 0)]) if (
            self.session and self.session.i) else 0.0

    def export(self):
        rows = [[r["i"], round(r["t"], 4), round(r["truth"], 4),
                 round(r["vision_raw"], 4), round(r["vision"], 4),
                 round(r["est"], 4), int(r["lost"]), int(r["outlier"]),
                 int(r["accepted"])] for r in self.log]
        export_csv(self, "纠偏逐帧数据.csv",
                   ["帧", "时间s", "真值m", "视觉原始m", "视觉零阶保持m",
                    "融合估计m", "丢失帧", "退化帧", "量测被采纳"], rows)
