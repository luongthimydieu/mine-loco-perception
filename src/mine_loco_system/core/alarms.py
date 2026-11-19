# -*- coding: utf-8 -*-
"""报警总线与全局工况状态。"""
import time
from collections import deque

from PySide6.QtCore import QObject, Signal

LEVELS = {"info": ("提示", "#3aa675"), "warn": ("预警", "#d8a028"),
          "alarm": ("报警", "#d84040"), "critical": ("紧急", "#ff5555")}


class AlarmBus(QObject):
    """集中报警总线：各模块发出，总览页与报警栏显示，可导出。"""
    alarm = Signal(dict)

    def __init__(self):
        super().__init__()
        self.records = deque(maxlen=500)

    def emit_alarm(self, source, level, message, advice=""):
        rec = {
            "time": time.strftime("%H:%M:%S"),
            "ts": time.time(),
            "source": source,
            "level": level,
            "level_name": LEVELS.get(level, ("提示", "#888"))[0],
            "message": message,
            "advice": advice,
        }
        self.records.appendleft(rec)
        self.alarm.emit(rec)

    def export_rows(self):
        return [dict(r) for r in self.records]


class WorkCondition(QObject):
    """全局工况：能见度 V、粉尘消光 β、车速。所有页面共享。"""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.V = 20.0            # 能见度 (m)
        self.speed = 3.0         # 车速 (m/s)
        self.obstacle_prob = 0.6
        self.scenes = {
            "无粉尘（检修硐室）": 200.0,
            "薄尘（V=50 m）": 50.0,
            "中尘（V=20 m）": 20.0,
            "浓尘（V=10 m）": 10.0,
        }

    @property
    def beta(self):
        return 3.0 / max(self.V, 0.1)

    def set_V(self, V):
        V = float(V)
        if abs(V - self.V) < 1e-9:
            self.V = V
            return
        self.V = V
        self.changed.emit()

    def set_scene(self, name):
        if name in self.scenes:
            self.V = self.scenes[name]
            self.changed.emit()
