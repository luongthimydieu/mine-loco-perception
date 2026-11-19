# -*- coding: utf-8 -*-
"""运维健康模块：轴承包络谱（E4a）+ 多源融合寿命预测（E4b）。

实现说明：所有控件在创建时显式指定父对象（QLabel(parent) 风格），
不依赖布局的延迟 reparent——PySide6 下布局子控件的父链建立时机在
部分场景不稳定，显式父对象可确保控件生命周期与页面一致。
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                               QPushButton, QGridLayout, QTabWidget, QWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QFrame)

from . import theme
from .widgets import Card, make_plot, export_csv
from ..core import health as H


class HealthPage(QFrame):
    ID = "health"
    TITLE = "④ 运维健康 · 状态维保"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.ctx = ctx
        self.hist = H.health_history()
        self._keep = []                      # 兜底持有全部中间对象引用
        self._build()
        self.on_snr(-6)
        self._emit_initial_advice()

    def _keepall(self, *objs):
        self._keep.extend(objs)

    # -------------------------------------------------------------- layout
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)
        tabs = QTabWidget(self)
        root.addWidget(tabs, 1)
        self._keepall(tabs, root)
        self._build_bearing(tabs)
        self._build_rul(tabs)

    # ---- E4a
    def _build_bearing(self, tabs):
        w = QWidget(tabs)
        root = QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        left = QVBoxLayout()
        root.addLayout(left, 2)
        c = Card("工况设置", w)
        gl = QGridLayout()
        t1 = QLabel("振动信噪比 SNR", c)
        gl.addWidget(t1, 0, 0)
        self.snr_lbl = QLabel("-6 dB", c)
        self.snr_lbl.setStyleSheet("font-weight:700; color:%s;" % theme.WARN)
        gl.addWidget(self.snr_lbl, 0, 2)
        self.snr_slider = QSlider(Qt.Horizontal, c)
        self.snr_slider.setRange(-15, 12)
        self.snr_slider.setValue(-6)
        gl.addWidget(self.snr_slider, 0, 1)
        self.snr_slider.valueChanged.connect(self.on_snr)
        c.body().addLayout(gl)
        self.det_lbl = QLabel("--", c)
        self.det_lbl.setStyleSheet("font-size:15px; font-weight:700;")
        self.det_lbl.setWordWrap(True)
        c.body().addWidget(self.det_lbl)
        self.ratio_lbl = QLabel("--", c)
        self.ratio_lbl.setObjectName("Dim")
        c.body().addWidget(self.ratio_lbl)
        b = QPushButton("导出包络谱数据 CSV", c)
        b.clicked.connect(self.export_spec)
        c.body().addWidget(b)
        left.addWidget(c)
        left.addStretch(1)
        info = Card("参数", w)
        il = QVBoxLayout()
        for t in ("轴转速 fr = 8 Hz（480 r/min）",
                  "外圈故障特征频率 BPFO = 5.4×fr = 43.2 Hz",
                  "采样率 12 kHz ｜ 带通 2~6 kHz",
                  "本底判据：包络谱中位数 + 3×1.4826×MAD",
                  "实验结论：SNR 低至 −9 dB 仍可靠检出"):
            lbl = QLabel("• " + t, info)
            lbl.setObjectName("Dim")
            lbl.setWordWrap(True)
            il.addWidget(lbl)
        il.addStretch(1)
        info.body().addLayout(il)
        info.setFixedWidth(300)
        left.addWidget(info)

        right = QVBoxLayout()
        root.addLayout(right, 8)
        self.wave_plot = make_plot("振动时域波形（1 s）", parent=w)
        right.addWidget(self.wave_plot, 3)
        self.spec_plot = make_plot("Hilbert 包络谱 与 BPFO 特征峰", parent=w)
        self.spec_plot.setXRange(0, 400)
        right.addWidget(self.spec_plot, 5)
        tabs.addTab(w, "轴承故障检测（包络谱）")
        self._keepall(w, root, left, right, gl, il, c, info, t1, b)

    # ---- E4b
    def _build_rul(self, tabs):
        w = QWidget(tabs)
        root = QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        left = QVBoxLayout()
        root.addLayout(left, 2)
        c = Card("健康评估结果", w)
        self.hi_lbl = QLabel("--", c)
        self.hi_lbl.setStyleSheet("font-size:15px; font-weight:700;")
        self.hi_lbl.setWordWrap(True)
        c.body().addWidget(self.hi_lbl)
        self.w_lbl = QLabel("--", c)
        self.w_lbl.setObjectName("Dim")
        self.w_lbl.setWordWrap(True)
        c.body().addWidget(self.w_lbl)
        b = QPushButton("导出退化与预测数据 CSV", c)
        b.clicked.connect(self.export_rul)
        c.body().addWidget(b)
        left.addWidget(c)
        left.addStretch(1)
        adv = Card("维保建议（自动生成）", w)
        al = QVBoxLayout()
        self.adv_lbl = QLabel("--", adv)
        self.adv_lbl.setWordWrap(True)
        al.addWidget(self.adv_lbl)
        adv.body().addLayout(al)
        left.addWidget(adv, 1)

        right = QVBoxLayout()
        root.addLayout(right, 8)
        self.hi_plot = make_plot("多源健康指标退化趋势 与 RUL 外推", parent=w)
        self.hi_plot.addLegend()
        right.addWidget(self.hi_plot, 5)
        self.rul_table = QTableWidget(0, 4, w)
        self.rul_table.setHorizontalHeaderLabels(
            ["预测时刻(天)", "预测剩余寿命(天)", "真实剩余寿命(天)", "相对误差(%)"])
        self.rul_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.rul_table.verticalHeader().setVisible(False)
        self.rul_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        right.addWidget(self.rul_table, 3)
        tabs.addTab(w, "剩余寿命预测与维保建议")
        self._keepall(w, root, left, right, al, c, adv, b)

    # ------------------------------------------------------------ handlers
    def on_snr(self, v):
        self.snr_lbl.setText("%d dB" % v)
        res = H.envelope_spectrum(v)
        t = np.arange(len(res["signal"])) / H.FS
        self.wave_plot.clear()
        self.wave_plot.plot(t[::6], res["signal"][::6],
                            pen=pg.mkPen("#7a8ba0", width=1))
        f, E = res["freq"], res["spectrum"]
        m = f <= 600
        self.spec_plot.clear()
        self.spec_plot.plot(f[m], E[m], pen=pg.mkPen(theme.ACCENT, width=1))
        self.spec_plot.addLine(y=res["floor"],
                               pen=pg.mkPen(theme.ALARM, style=Qt.DashLine))
        bpfo_line = pg.InfiniteLine(pos=H.BPFO, angle=90,
                                    pen=pg.mkPen(theme.OK, style=Qt.DotLine))
        self.spec_plot.addItem(bpfo_line)
        band = (f > H.BPFO * 0.9) & (f < H.BPFO * 1.1)
        peak_x = f[band][int(np.argmax(E[band]))]
        sp = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(theme.OK))
        self.spec_plot.addItem(sp)
        sp.setData([peak_x], [res["peak"]])
        det = "检出" if res["detected"] else "未检出"
        color = theme.OK if res["detected"] else theme.ALARM
        self.det_lbl.setText(
            "SNR = %d dB：BPFO=%.1f Hz 处 %s（谱峰/本底 = %.2f）"
            % (v, H.BPFO, det, res["ratio"]))
        self.det_lbl.setStyleSheet(
            "font-size:15px; font-weight:700; color:%s;" % color)
        self.ratio_lbl.setText(
            "谱峰 %.1f ｜ 噪声本底阈值 %.1f ｜ 实验结论：−9 dB 下限（6 档 SNR 全部检出）"
            % (res["peak"], res["floor"]))
        self._last_spec = res
        if not res["detected"]:
            self.ctx.alarms.emit_alarm(
                "运维健康", "warn",
                "SNR=%d dB：轴承故障特征未检出，建议降低监测阈值并复核传感器安装" % v)

    def _emit_initial_advice(self):
        fail_idx = self.hist["fail_idx"]
        rows = H.rul_table(self.hist["fused"], fail_idx)
        last = rows[-1] if rows else None
        rul = last["预测剩余寿命(天)"] if last else None
        ratio = getattr(self, "_last_spec", {}).get("ratio")
        self._update_advice(rul, ratio)

    def _update_advice(self, rul, ratio):
        from ..core.health import maintenance_advice
        adv = maintenance_advice(rul, ratio)
        txt = "\n".join("【%s】%s" % (lv, msg) for lv, msg in adv)
        self.adv_lbl.setText(txt)
        for lv, msg in adv:
            level = {"紧急": "critical", "重要": "alarm",
                     "关注": "warn"}.get(lv, "info")
            self.ctx.alarms.emit_alarm("运维健康", level, msg)

    def on_show(self):
        """切换到本页时刷新 RUL 表与退化曲线。"""
        fail_idx = self.hist["fail_idx"]
        rows = H.rul_table(self.hist["fused"], fail_idx)
        self.rul_table.setRowCount(0)
        for r in rows:
            i = self.rul_table.rowCount()
            self.rul_table.insertRow(i)
            for j, k in enumerate(("预测时刻(天)", "预测剩余寿命(天)",
                                   "真实剩余寿命(天)", "相对误差(%)")):
                self.rul_table.setItem(i, j, QTableWidgetItem(str(r[k])))
        self.hi_plot.clear()
        days = H.DAYS
        fused = self.hist["fused"]
        cut = int(fail_idx * 0.85)
        rp, ah, bh = H.rul_predict(fused, cut)
        fit_x = np.arange(0, fail_idx + 40)
        fit_y = ah * np.exp(bh * fit_x)
        self.hi_plot.addLine(y=1.0, pen=pg.mkPen(theme.ALARM, style=Qt.DashLine))
        self.hi_plot.plot(days, self.hist["health"], pen=None,
                          symbol="o", symbolSize=3,
                          symbolBrush="#557090", name="健康指标观测")
        self.hi_plot.plot(days, self.hist["hi_true"],
                          pen=pg.mkPen("#7a8ba0", width=1, style=Qt.DashLine),
                          name="真值退化")
        self.hi_plot.plot(fit_x, np.clip(fit_y, 0, 1.6),
                          pen=pg.mkPen(theme.ACCENT, width=2),
                          name="RUL 外推（第 %d 天预测）" % cut)
        vline = pg.InfiniteLine(pos=cut, angle=90,
                                pen=pg.mkPen(theme.WARN, style=Qt.DotLine))
        self.hi_plot.addItem(vline)
        self.hi_plot.setYRange(0, 1.35)
        wtxt = " ｜ ".join("%s %.1f%%" % (k, v * 100)
                           for k, v in self.hist["weights"].items())
        self.w_lbl.setText("融合权重（最小二乘，标定于失效前60%%）：%s" % wtxt)
        if rp is not None:
            self.hi_lbl.setText(
                "第 %d 天预测：剩余寿命 %.0f 天（实际剩余 %d 天，相对误差 %.1f%%）"
                % (cut, rp, fail_idx - cut,
                   abs(rp - (fail_idx - cut)) / (fail_idx - cut) * 100))
        self._update_advice(rp if rp is not None else None, None)

    def export_spec(self):
        res = getattr(self, "_last_spec", None)
        if not res:
            return
        f, E = res["freq"], res["spectrum"]
        m = (f > 0) & (f <= 800)
        rows = [[round(float(fi), 2), round(float(ei), 2)]
                for fi, ei in zip(f[m], E[m])]
        export_csv(self, "包络谱.csv", ["频率Hz", "幅值"], rows)

    def export_rul(self):
        fail_idx = self.hist["fail_idx"]
        rows = [[r["预测时刻(天)"], r["预测剩余寿命(天)"], r["真实剩余寿命(天)"],
                 r["相对误差(%)"]] for r in H.rul_table(self.hist["fused"],
                                                        fail_idx)]
        export_csv(self, "寿命预测.csv",
                   ["预测时刻(天)", "预测剩余寿命(天)", "真实剩余寿命(天)",
                    "相对误差(%)"], rows)
