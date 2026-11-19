# -*- coding: utf-8 -*-
"""历史回放：读取导出的逐帧 CSV，回放曲线与判决时间线。"""
import csv
import os

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QFileDialog, QTableWidget, QTableWidgetItem,
                               QHeaderView, QAbstractItemView, QFrame)

from . import theme
from .widgets import make_plot
from ..paths import ASSETS_DIR


class ReplayPage(QFrame):
    ID = "replay"
    TITLE = "历史数据回放"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.ctx = ctx
        self.rows = []
        self.pos = 0
        self._keep = []
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.setInterval(33)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)
        bar = QHBoxLayout()
        b1 = QPushButton("加载导出的逐帧数据 CSV…")
        b1.setObjectName("Primary")
        b1.clicked.connect(self.pick)
        self.demo_btn = QPushButton("▶ 回放")
        self.demo_btn.clicked.connect(self.toggle)
        self.demo_btn.setEnabled(False)
        bar.addWidget(b1)
        bar.addWidget(self.demo_btn)
        self.file_lbl = QLabel("支持：纠偏逐帧数据.csv / 雷达逐帧判决.csv / "
                               "检测结果.csv（各模块“导出”生成）")
        self.file_lbl.setObjectName("Dim")
        bar.addWidget(self.file_lbl, 1)
        root.addLayout(bar)

        self.plot = make_plot("历史数据回放")
        self.plot.addLegend()
        root.addWidget(self.plot, 5)
        self.c1 = self.plot.plot(pen=pg.mkPen(theme.WARN, width=1),
                                 name="纯视觉")
        self.c2 = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=2),
                                 name="融合估计")
        vline = pg.InfiniteLine(angle=90, movable=True,
                                pen=pg.mkPen(theme.OK))
        self.plot.addItem(vline)
        self.vline = vline

        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 4)
        self._keep += [bar, root]

    def on_show(self):
        if not self.rows:
            demo = str(ASSETS_DIR / "sample" / "correction_frames.csv")
            if os.path.exists(demo):
                self._load_csv(demo)

    def pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 CSV", "",
                                           "CSV (*.csv)")
        if p:
            self._load_csv(p)

    def _load_csv(self, p):
        with open(p, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        self.header = rows[0]
        self.rows = rows[1:]
        self.file_lbl.setText("%s（%d 行）" % (os.path.basename(p),
                                              len(self.rows)))
        self.table.setRowCount(0)
        self.table.setColumnCount(len(self.header))
        self.table.setHorizontalHeaderLabels(self.header)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.demo_btn.setEnabled(True)
        self.pos = 0
        self.c1.setData([], [])
        self.c2.setData([], [])
        self.c_truth = None
        # 判断列
        self._cols = {h: i for i, h in enumerate(self.header)}
        self._is_fusion = "真值m" in self._cols
        self._is_lidar = "判决" in self._cols and "有障碍" in self._cols
        # 加载即渲染全量曲线 + 首屏表格
        self._render(len(self.rows))
        self.pos = 0

    def _render(self, n):
        xs = list(range(n))
        try:
            if self._is_fusion and n > 1:
                truth = [float(r[self._cols["真值m"]]) for r in self.rows[:n]]
                est = [float(r[self._cols["融合估计m"]]) for r in self.rows[:n]]
                vis = [float(r[self._cols["视觉零阶保持m"]]) for r in self.rows[:n]]
                self.c1.setData(xs, vis)
                self.c2.setData(xs, est)
                self.plot.setYRange(-0.25, 0.25)
                self.c_truth = getattr(self, "c_truth", None)
                if self.c_truth is None:
                    self.c_truth = self.plot.plot(pen=pg.mkPen("#e8eef5",
                                                               width=2),
                                                  name="真值")
                self.c_truth.setData(xs, truth)
            elif self._is_lidar and n > 1:
                verdicts = [r[self._cols["判决"]] for r in self.rows[:n]]
                self.c1.setData(xs, [0] * n)
                self.c2.setData(xs, [1 if v == "hit" else
                                     (0.5 if v == "false" else 0.3)
                                     for v in verdicts])
                self.plot.setYRange(0, 1.2)
        except (ValueError, KeyError):
            pass
        self.vline.setPos(n)
        # 表格滚动窗口
        show = self.rows[max(0, n - 12):n]
        self.table.setRowCount(0)
        for r in show:
            i = self.table.rowCount()
            self.table.insertRow(i)
            for j, v in enumerate(r):
                self.table.setItem(i, j, QTableWidgetItem(v))

    def toggle(self):
        if not self.rows:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.demo_btn.setText("▶ 回放")
        else:
            self._timer.start()
            self.demo_btn.setText("⏸ 暂停")

    def tick(self):
        if not self.rows:
            return
        self.pos = (self.pos + 3) % (len(self.rows) + 1)
        self._render(self.pos)
