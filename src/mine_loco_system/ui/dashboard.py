# -*- coding: utf-8 -*-
"""总览仪表盘：系统状态、KPI、报警中心。"""
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QFrame,
                               QComboBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QAbstractItemView)

from . import theme
from .widgets import Card, KpiCard, StatusDot, make_plot
from ..core.lidar import reliable_range


class Dashboard(QFrame):
    ID = "dashboard"
    TITLE = "系统总览"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.ctx = ctx          # AppContext: state/alarms/app
        self._keep = []
        self.err_series = {"t": [], "vision": [], "est": []}
        self._build()
        self.ctx.alarms.alarm.connect(self.on_alarm)
        self.ctx.state.changed.connect(self.on_condition)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_live)
        self._timer.start(500)
        self.on_condition()

    # -------------------------------------------------------------- layout
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # 左列（弹性）
        left = QVBoxLayout()
        left.setSpacing(12)
        root.addLayout(left, 7)

        # 工况条
        cond = Card("当前工况")
        crow = QHBoxLayout()
        crow.addWidget(QLabel("场景："))
        self.scene_box = QComboBox()
        self.scene_box.addItems(list(self.ctx.state.scenes.keys()))
        cur = min(self.ctx.state.scenes.items(), key=lambda kv: abs(kv[1] - self.ctx.state.V))[0]
        self.scene_box.setCurrentText(cur)
        self.scene_box.currentTextChanged.connect(self._on_scene)
        crow.addWidget(self.scene_box)
        crow.addWidget(QLabel("车速："))
        self.speed_lbl = QLabel("--")
        self.speed_lbl.setStyleSheet("font-weight:700; color:%s;" % theme.ACCENT)
        crow.addWidget(self.speed_lbl)
        crow.addStretch(1)
        self.vis_lbl = QLabel("能见度 -- m ｜ β -- /m")
        self.vis_lbl.setStyleSheet("font-weight:700; color:%s;" % theme.WARN)
        crow.addWidget(self.vis_lbl)
        cond.body().addLayout(crow)
        left.addWidget(cond)

        # KPI 四卡
        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        self.kpi_rmse = KpiCard(
            "横向偏移 RMSE（高粉尘）", "-- m", "设计目标 ≤20 mm ｜ 纯视觉 0.08 m",
            theme.OK)
        self.kpi_rate = KpiCard(
            "逐帧纠偏成功率", "-- %", "判据 |误差| ≤ 20 mm ｜ 纯视觉 65.6%", theme.OK)
        self.kpi_lidar = KpiCard(
            "雷达可靠检出距离", "-- m", "双程消光边界（β=3/V）", theme.WARN)
        self.kpi_rul = KpiCard(
            "剩余寿命预测误差", "11.47 %", "多源融合 ｜ 单一温升 11.85%", theme.FG)
        for k in (self.kpi_rmse, self.kpi_rate, self.kpi_lidar, self.kpi_rul):
            kpis.addWidget(k)
        left.addLayout(kpis)

        # 中部：运行状态 + 纠偏曲线
        mid = QHBoxLayout()
        mid.setSpacing(12)
        status_card = Card("行走机构运行状态")
        sl = QVBoxLayout()
        self.dot_perc = StatusDot("视觉感知")
        self.dot_corr = StatusDot("融合纠偏")
        self.dot_health = StatusDot("运维健康")
        self.dot_lidar = StatusDot("激光雷达")
        for d in (self.dot_perc, self.dot_corr, self.dot_lidar, self.dot_health):
            sl.addWidget(d)
        sl.addSpacing(8)
        self.pos_lbl = QLabel("里程：-- m ｜ 偏移：-- mm")
        self.pos_lbl.setObjectName("Dim")
        sl.addWidget(self.pos_lbl)
        self.speed2_lbl = QLabel("车速：-- m/s（约 -- km/h）")
        self.speed2_lbl.setObjectName("Dim")
        sl.addWidget(self.speed2_lbl)
        sl.addStretch(1)
        status_card.body().addLayout(sl)
        status_card.setFixedWidth(230)
        mid.addWidget(status_card)

        self.err_plot = make_plot("横向偏移估计误差（实时）")
        self.err_plot.addLegend()
        self.c_vis = self.err_plot.plot(pen=pg.mkPen(theme.WARN, width=1),
                                        name="纯视觉")
        self.c_est = self.err_plot.plot(pen=pg.mkPen(theme.ACCENT, width=2),
                                        name="融合纠偏")
        mid.addWidget(self.err_plot, 1)
        left.addLayout(mid, 1)

        # 右列：报警中心
        right = Card("报警与维保建议")
        rl = QVBoxLayout()
        self.alarm_table = QTableWidget(0, 4)
        self.alarm_table.setHorizontalHeaderLabels(["时间", "级别", "来源", "内容"])
        self.alarm_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        self.alarm_table.verticalHeader().setVisible(False)
        self.alarm_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.alarm_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rl.addWidget(self.alarm_table)
        right.body().addLayout(rl, 1)
        right.setFixedWidth(430)
        root.addWidget(right, 4)
        self._keep += [cond, crow, left, right, rl, kpis, mid, sl, status_card]

    # ------------------------------------------------------------ handlers
    def _on_scene(self, name):
        self.ctx.state.set_scene(name)
        self.ctx.alarms.emit_alarm(
            "工况", "info", f"场景切换：{name}（能见度 {self.ctx.state.V:.0f} m）")

    def on_condition(self):
        s = self.ctx.state
        self.vis_lbl.setText(f"能见度 {s.V:.0f} m ｜ β {s.beta:.3f} /m")
        # 子系统状态灯（激光雷达按理论可靠距离分级）
        r = reliable_range(s.V)
        self.dot_perc.set_status("ok", "视觉感知")
        self.dot_corr.set_status("ok", "融合纠偏")
        self.dot_lidar.set_status("ok" if r >= 10 else "warn", "激光雷达")
        self.dot_health.set_status("ok", "运维健康")
        self.speed_lbl.setText(f"{s.speed:.1f} m/s")
        self.speed2_lbl.setText(
            f"车速：{s.speed:.1f} m/s（约 {s.speed * 3.6:.0f} km/h）")
        # KPI：雷达理论距离
        r = reliable_range(s.V)
        warn = r < 10
        self.kpi_lidar.set(
            f"{r:.1f} m",
            "双程消光边界（β=%.3f /m）%s" % (
                s.beta, "⚠ 与制动距离同量级" if warn else ""),
            theme.ALARM if warn else theme.WARN)

    def attach_fusion(self, fusion_module):
        """由主窗口注入纠偏模块的实时统计。"""
        self.fusion_module = fusion_module

    def _refresh_live(self):
        fm = getattr(self, "fusion_module", None)
        if fm is None:
            return
        summ = fm.live_summary()
        if not summ:
            return
        self.kpi_rmse.set(f"{summ['RMSE_本方案_m']*100:.1f} cm",
                          f"纯视觉 {summ['RMSE_纯视觉_m']*100:.1f} cm ｜ 目标 ≤2 cm")
        self.kpi_rate.set(f"{summ['成功率_本方案_%']:.1f} %",
                          f"纯视觉 {summ['成功率_纯视觉_%']:.1f} %")
        self.dot_corr.set_status(
            "ok" if summ["成功率_本方案_%"] > 70 else "warn", "融合纠偏")
        data = fm.error_series()
        if data:
            self.c_vis.setData(data["t"], data["vision"])
            self.c_est.setData(data["t"], data["est"])
        self.pos_lbl.setText(
            f"里程：{fm.odometer():.0f} m ｜ "
            f"偏移：{fm.last_offset()*1000:+.0f} mm")

    def on_alarm(self, rec):
        self.alarm_table.insertRow(0)
        vals = [rec["time"], rec["level_name"], rec["source"], rec["message"]]
        colors = {"info": theme.FG_DIM, "warn": theme.WARN,
                  "alarm": theme.ALARM, "critical": "#ff6666"}
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if c == 1:
                fg = colors.get(rec["level"], theme.FG)
                it.setForeground(pg.mkColor(fg))
            if c == 3:
                it.setToolTip(rec.get("advice", "") or v)
            self.alarm_table.setItem(0, c, it)
        if self.alarm_table.rowCount() > 200:
            self.alarm_table.removeRow(self.alarm_table.rowCount() - 1)
