# -*- coding: utf-8 -*-
"""主窗口：侧边导航 + 页面容器 + 全局状态/报警总线。"""
import os
import time

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                               QLabel, QPushButton, QButtonGroup, QStackedWidget,
                               QStatusBar, QFrame, QApplication)

from .dashboard import Dashboard
from .perception import PerceptionPage
from .correction import CorrectionPage
from .lidar_page import LidarPage
from .health_page import HealthPage
from .replay import ReplayPage
from ..core.alarms import AlarmBus, WorkCondition

APP_NAME = "井下电机车智能感知与运维系统"
VERSION = "v1.0"


class AppContext:
    """跨页面共享：全局工况 + 报警总线 + 主窗引用。"""

    def __init__(self, win):
        self.win = win
        self.state = WorkCondition()
        self.alarms = AlarmBus()

    def toast(self, msg):
        self.win.statusBar().showMessage(msg, 4000)


class MainWindow(QMainWindow):
    PAGES = (Dashboard, PerceptionPage, CorrectionPage, LidarPage,
             HealthPage, ReplayPage)

    def __init__(self, screenshot_dir=None, record_path=None,
                 demo_duration=24):
        super().__init__()
        self.setWindowTitle("%s %s" % (APP_NAME, VERSION))
        self.resize(1600, 940)
        self.ctx = AppContext(self)
        self.screenshot_dir = screenshot_dir
        self.record_path = record_path
        self.demo_duration = demo_duration
        self._recorder = None

        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 侧边栏
        side = QFrame()
        side.setObjectName("Side")
        side.setFixedWidth(190)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 18, 12, 12)
        logo = QLabel("⛏ " + APP_NAME)
        logo.setObjectName("Title")
        logo.setWordWrap(True)
        sl.addWidget(logo)
        sub = QLabel("物理引导域适应 ｜ %s" % VERSION)
        sub.setObjectName("Dim")
        sl.addWidget(sub)
        sl.addSpacing(14)
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.nav_btns = []
        for i, cls in enumerate(self.PAGES):
            b = QPushButton("%d  %s" % (i, cls.TITLE) if False else cls.TITLE)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("text-align:left; padding:9px 12px;")
            b.clicked.connect(lambda _=False, k=i: self.goto(k))
            self.btn_group.addButton(b, i)
            sl.addWidget(b)
            self.nav_btns.append(b)
        sl.addStretch(1)
        self.clock_lbl = QLabel("--")
        self.clock_lbl.setObjectName("Dim")
        sl.addWidget(self.clock_lbl)
        root.addWidget(side)

        # 页面容器
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.pages = []
        for cls in self.PAGES:
            p = cls(self.ctx)
            self.stack.addWidget(p)
            self.pages.append(p)

        # 状态栏
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.cond_lbl = QLabel("--")
        sb.addPermanentWidget(self.cond_lbl)
        self.ctx.state.changed.connect(self._update_cond)
        self._update_cond()
        self.ctx.alarms.emit_alarm("系统", "info", "系统启动完成，各模块自检通过")

        self.pages[0].attach_fusion(self.pages[2])

        self.nav_btns[0].setChecked(True)
        self.goto(0)

        # 时钟
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick_clock)
        self._clock.start(1000)
        self._tick_clock()

    # ------------------------------------------------------------ helpers
    def goto(self, i):
        self.stack.setCurrentIndex(i)
        p = self.pages[i]
        if hasattr(p, "on_show"):
            p.on_show()

    def _update_cond(self):
        s = self.ctx.state
        self.cond_lbl.setText(
            "能见度 %.0f m ｜ β %.3f /m ｜ 车速 %.1f m/s"
            % (s.V, s.beta, s.speed))

    def _tick_clock(self):
        self.clock_lbl.setText(time.strftime("%Y-%m-%d %H:%M:%S"))

    # ------------------------------------------------------------ 截图
    def take_screenshots(self, out=None):
        out = out or self.screenshot_dir
        os.makedirs(out, exist_ok=True)
        names = []
        for i, p in enumerate(self.pages):
            self.goto(i)
            for _ in range(8):                 # 让定时器跑几拍
                QApplication.processEvents()
                time.sleep(0.05)
            QApplication.processEvents()
            name = "%02d_%s.png" % (i, p.ID)
            pm = self.grab()
            pm.save(os.path.join(out, name))
            names.append(name)
            print("saved", name, flush=True)
        return names

    # ------------------------------------------------------------ 演示模式
    def run_demo(self):
        """自动演示：按脚本切换场景与页面（供录像）。"""
        self._demo_step = 0
        self._demo_timer = QTimer(self)
        self._demo_timer.timeout.connect(self._demo_tick)
        self._demo_timer.setInterval(int(self.demo_duration * 1000 / 16))
        self._demo_plan = [
            (0, "总览：系统自检、三子系统状态灯就绪"),
            (0, None),
            (1, "视觉感知：加载 ExDark 低照度图像，无粉尘工况基线检测"),
            (1, None),
            (1, "视觉感知：切换浓尘工况（V=10 m），观察对比度塌缩与检测衰减"),
            (1, None),
            (2, "融合纠偏：β=0.30 高粉尘，纯视觉 vs 融合估计实时对比"),
            (2, None),
            (3, "激光雷达：无粉尘基准扫描 → β 逐档加大，回波被双程消光压制"),
            (3, None),
            (4, "运维健康：−6 dB 包络谱检出 BPFO 谱峰；RUL 外推与维保建议"),
            (4, None),
            (0, "回到总览：报警与维保建议汇总"),
            (0, None),
            (5, "历史回放：导出数据复现"),
            (5, None),
        ]
        self._demo_timer.start()

    def _demo_tick(self):
        i = self._demo_step
        if i >= len(self._demo_plan):
            self._demo_timer.stop()
            self._stop_recording()
            from PySide6.QtCore import QTimer as _Q
            _Q.singleShot(800, QApplication.instance().quit)
            return
        page_i, note = self._demo_plan[i]
        self.goto(page_i)
        p = self.pages[page_i]
        if hasattr(p, "demo_step") and note is not None:
            p.demo_step(i)
        if page_i == 1 and note and "浓尘" in note:
            p.v_slider.setValue(10)
            p.on_condition()
        if page_i == 3 and note:
            vals = [0, 10, 20, 30, 40]
            p.b_slider.setValue(vals[min(i // 1, len(vals) - 1) % len(vals)])
            p.on_beta()
        if note:
            self.statusBar().showMessage("演示：%s" % note, 3000)
        self._demo_step += 1

    # ------------------------------------------------------------ 录像
    def start_recording(self, path=None, fps=5):
        path = path or self.record_path
        if not path:
            return False
        self._rec_fps = fps
        self._rec_path = path
        self._rec_frames = []
        self._grab_timer = QTimer(self)
        self._grab_timer.timeout.connect(self._grab_frame)
        self._grab_timer.setInterval(int(1000 / fps))
        self._grab_timer.start()
        print("[record] ->", path, flush=True)
        return True

    def _grab_frame(self):
        pm = self.grab()
        from PySide6.QtGui import QImage as _QImg
        img = pm.toImage().convertToFormat(_QImg.Format.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, img.bytesPerLine())
        arr = arr[:, :w * 3].reshape(h, w, 3)
        self._rec_frames.append(arr[:, :, ::-1].copy())
        if len(self._rec_frames) % 20 == 0:
            print("[record] %d frames" % len(self._rec_frames), flush=True)

    def _stop_recording(self):
        if self._recorder is not None:
            return
        if not getattr(self, "_rec_frames", None):
            return
        self._grab_timer.stop()
        import cv2
        h, w = self._rec_frames[0].shape[:2]
        vw = cv2.VideoWriter(self._rec_path,
                             cv2.VideoWriter_fourcc(*"mp4v"), self._rec_fps,
                             (w, h))
        for f in self._rec_frames:
            vw.write(f)
        vw.release()
        print("[record] saved %s (%d frames)" % (self._rec_path,
                                                 len(self._rec_frames)),
              flush=True)
