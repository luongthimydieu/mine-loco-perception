# -*- coding: utf-8 -*-
"""深色工业监控主题（矿井调度中心风格）。"""

BG = "#12161c"          # 主背景
BG_PANEL = "#1a212b"    # 面板
BG_CARD = "#202a37"     # 卡片
BORDER = "#2c3a4d"
FG = "#d7e1ee"          # 主文字
FG_DIM = "#8a9bb0"      # 次文字
ACCENT = "#3da5f5"      # 主色（矿灯蓝）
OK = "#3aa675"
WARN = "#d8a028"
ALARM = "#d84040"

QSS = f"""
* {{
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: {FG};
}}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: 17px; font-weight: 600; }}
QLabel#H1 {{ font-size: 15px; font-weight: 600; color: {ACCENT}; }}
QLabel#Dim {{ color: {FG_DIM}; }}
QLabel#KpiValue {{ font-size: 26px; font-weight: 700; }}
QLabel#KpiName {{ color: {FG_DIM}; font-size: 12px; }}

QFrame#Card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#Side {{ background: {BG_PANEL}; border: none; }}

QPushButton {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 6px 14px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton#Primary {{
    background: {ACCENT}; color: #06121f; font-weight: 600; border: none;
}}
QPushButton#Primary:hover {{ background: #5cb7f8; }}
QPushButton#Danger {{ background: {ALARM}; color: white; border: none; }}

QListWidget, QTreeWidget, QTableWidget {{
    background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 6px;
}}
QHeaderView::section {{
    background: {BG_CARD}; border: none; padding: 5px;
}}
QTableView::item, QTreeWidget::item {{ padding: 4px; }}

QSlider::groove:horizontal {{
    height: 5px; background: {BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; margin: -5px 0; border-radius: 7px;
    background: {ACCENT};
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

QComboBox {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 5px 10px;
}}
QComboBox QAbstractItemView {{
    background: {BG_PANEL}; selection-background-color: {ACCENT};
}}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; }}
QTabBar::tab {{
    background: {BG_PANEL}; padding: 7px 16px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    color: {FG_DIM};
}}
QTabBar::tab:selected {{ background: {BG_CARD}; color: {ACCENT}; }}

QProgressBar {{
    background: {BORDER}; border: none; border-radius: 4px;
    height: 8px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QStatusBar {{ background: {BG_PANEL}; color: {FG_DIM}; }}
QScrollBar:vertical {{
    background: {BG_PANEL}; width: 9px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 30px;
}}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QToolTip {{
    background: {BG_CARD}; color: {FG}; border: 1px solid {BORDER};
}}
"""
