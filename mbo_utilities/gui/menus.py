# from qtpy import QtGui
from qtpy.QtWidgets import QAction, QMenu
# from pkg_resources import iter_entry_points
from .io import load_dialog, load_dialog_folder


def mainmenu(parent):
    # --------------- MENU BAR --------------------------
    # # run suite2p from scratch
    # runS2P = QAction("&Run suite2p", parent)
    # runS2P.setShortcut("Ctrl+R")
    # runS2P.triggered.connect(lambda: run_suite2p(parent))
    # parent.addAction(runS2P)

    # load processed data
    loadProc = QAction("&Load Single Z-Plane", parent)
    loadProc.setShortcut("Ctrl+L")
    loadProc.triggered.connect(lambda: load_dialog(parent))
    parent.addAction(loadProc)

    # load folder of processed data
    loadFolder = QAction("Load &Folder of Z-Planes", parent)
    loadFolder.setShortcut("Ctrl+F")
    loadFolder.triggered.connect(lambda: load_dialog_folder(parent))
    parent.addAction(loadFolder)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(loadProc)
    file_menu.addAction(loadFolder)