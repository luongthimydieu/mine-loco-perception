# -*- coding: utf-8 -*-
"""PyInstaller entry point for the desktop app."""
import multiprocessing

from mine_loco_system.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
