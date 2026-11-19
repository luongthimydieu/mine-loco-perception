# -*- coding: utf-8 -*-
"""公共控件：卡片、状态灯、KPI、曲线页。"""
import csv
import os

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QLabel, QVBoxLayout, QHBoxLayout,
                               QFileDialog, QMessageBox)

from . import theme

pg.setConfigOptions(background=theme.BG_PANEL, foreground=theme.FG_DIM,
                    antialias=True)


class Card(QFrame):
    """带标题的面板卡片。"""

    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(12, 10, 12, 12)
        self._lay.setSpacing(8)
        if title:
            t = QLabel(title)
            t.setObjectName("H1")
            self._lay.addWidget(t)

    def body(self):
        return self._lay

    def add(self, w):
        self._lay.addWidget(w)
        return w


class StatusDot(QLabel):
    """子系统状态灯。"""

    COLOR = {"ok": theme.OK, "warn": theme.WARN, "alarm": theme.ALARM,
             "off": "#556"}

    def __init__(self, name="系统"):
        super().__init__(name)
        self._name = name
        self.set_status("off")

    def set_status(self, key, text=None):
        c = self.COLOR.get(key, "#556")
        self.setText(f"● {text or self._name}")
        self.setStyleSheet(f"color:{c}; font-weight:600;")


class KpiCard(QFrame):
    """KPI 指标卡：名称 + 数值 + 对比说明。"""

    def __init__(self, name, value="--", note="", color=theme.FG, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        self.name_lbl = QLabel(name)
        self.name_lbl.setObjectName("KpiName")
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("KpiValue")
        self.value_lbl.setStyleSheet(f"color:{color};")
        self.note_lbl = QLabel(note)
        self.note_lbl.setObjectName("Dim")
        self.note_lbl.setWordWrap(True)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.value_lbl)
        lay.addWidget(self.note_lbl)

    def set(self, value, note=None, color=None):
        self.value_lbl.setText(str(value))
        if color:
            self.value_lbl.setStyleSheet(f"color:{color};")
        if note is not None:
            self.note_lbl.setText(note)


def make_plot(title=None, height=None, parent=None):
    """统一风格的 pyqtgraph PlotWidget。"""
    p = pg.PlotWidget(parent=parent)
    p.showGrid(x=True, y=True, alpha=0.25)
    p.getAxis("bottom").setPen(pg.mkPen(theme.BORDER))
    p.getAxis("left").setPen(pg.mkPen(theme.BORDER))
    if title:
        p.setTitle(title, color=theme.FG, size="12pt")
    if height:
        p.setMinimumHeight(height)
    p.addLegend(offset=(8, 8), labelTextSize="10px")
    return p


def export_csv(parent, default_name, header, rows):
    """通用试验数据导出（CSV，UTF-8 BOM 便于 Excel 打开）。"""
    path, _ = QFileDialog.getSaveFileName(
        parent, "导出试验数据", default_name, "CSV (*.csv)")
    if not path:
        return False
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        QMessageBox.information(parent, "导出完成", f"已导出 {len(rows)} 行 →\n{path}")
        return True
    except Exception as e:
        QMessageBox.warning(parent, "导出失败", str(e))
        return False
