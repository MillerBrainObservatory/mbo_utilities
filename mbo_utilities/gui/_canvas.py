"""custom qt canvas with icon support."""
from PySide6.QtWidgets import QApplication, QHBoxLayout
from PySide6.QtGui import QIcon
from rendercanvas.qt import (
    QRenderWidget,
    WrapperRenderCanvas,
    QtWidgets,
    WA_DeleteOnClose,
    loop,
)


def _get_icon():
    """get icon, returns None if not found."""
    try:
        from mbo_utilities.gui._setup import _get_icon_path
        path = _get_icon_path()
        if path:
            return QIcon(str(path))
    except Exception:
        pass
    return None


class MboRenderCanvas(WrapperRenderCanvas, QtWidgets.QWidget):
    """QRenderCanvas subclass that sets window icon before showing."""

    def __init__(self, parent=None, **kwargs):
        loop._rc_init()
        super().__init__(parent)

        self._subwidget = QRenderWidget(self, **kwargs)

        self.setAttribute(WA_DeleteOnClose, True)
        self.setMouseTracking(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self._subwidget)

        # set icon before show
        icon = _get_icon()
        if icon:
            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app:
                app.setWindowIcon(icon)

        self.show()
        self._final_canvas_init()

    def update(self):
        self._subwidget.request_draw()
        super().update()

    def closeEvent(self, event):
        self._subwidget.closeEvent(event)


# alias for consistency
RenderCanvas = MboRenderCanvas
