# -*- coding: utf-8 -*-
"""井下电机车智能感知与运维系统 —— 程序入口。

用法：
  python main.py                     # 正常启动
  python main.py --screenshots DIR   # 自动遍历各页面截图（系统界面截图）
  python main.py --demo --record OUT.mp4 [--duration 24]
                                     # 自动演示并录屏（演示录像）
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", metavar="DIR")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--record", metavar="MP4")
    ap.add_argument("--duration", type=float, default=24)
    args = ap.parse_args()

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    from .ui.theme import QSS
    app.setStyleSheet(QSS)

    from .ui.app import MainWindow
    win = MainWindow(screenshot_dir=args.screenshots,
                     record_path=args.record, demo_duration=args.duration)
    if args.record:
        win.resize(1280, 720)
    win.show()

    if args.screenshots:
        for _ in range(10):
            from PySide6.QtWidgets import QApplication as _A
            _A.processEvents()
        win.take_screenshots()
        return 0

    if args.demo:
        from PySide6.QtCore import QTimer
        if args.record:
            win.start_recording()

        def _start():
            win.run_demo()
        QTimer.singleShot(1500, _start)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
