import sys

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtSvg import QSvgWidget
from PySide2.QtCore import QByteArray


class DisplaySVG(QtWidgets.QWidget):
    # A simple SVG display.
    def __init__(self, svg: str, parent=None):
        super().__init__(parent)

        width_ind = svg.find('width')
        start_ind = svg.find('\"', width_ind + 1)
        end_ind = svg.find('\"', start_ind + 1)
        width = int(svg[start_ind + 1:end_ind - 2]) * 5

        height_ind = svg.find('height')
        start_ind = svg.find('\"', height_ind + 1)
        end_ind = svg.find('\"', start_ind + 1)
        height = int(svg[start_ind + 1:end_ind - 2]) * 5

        self.setWindowTitle("Automaton")
        act = QtWidgets.QAction("Close", self)
        act.setShortcuts([QtGui.QKeySequence(QtCore.Qt.Key_Escape)])
        act.triggered.connect(self.close)
        self.addAction(act)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(QtCore.QRect(0, 0, width, height))
        self.widgetSvg.load(QByteArray(svg.encode('utf-8')))

        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidget(self.widgetSvg)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.resize(width, height)
        self.verticalLayout.addWidget(self.scrollArea)


def show_svg(svg):
    qt_app = QtWidgets.QApplication(sys.argv)
    disp = DisplaySVG(svg)
    disp.show()
    qt_app.exec_()
