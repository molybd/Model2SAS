# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'subwindow_htmlview.ui'
##
## Created by: Qt User Interface Compiler version 6.5.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QGridLayout, QPushButton, QSizePolicy,
    QSpacerItem, QWidget)

class Ui_subWindow_html_view(object):
    def setupUi(self, subWindow_html_view):
        if not subWindow_html_view.objectName():
            subWindow_html_view.setObjectName(u"subWindow_html_view")
        subWindow_html_view.resize(500, 400)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(subWindow_html_view.sizePolicy().hasHeightForWidth())
        subWindow_html_view.setSizePolicy(sizePolicy)
        subWindow_html_view.setMinimumSize(QSize(400, 300))
        self.gridLayout = QGridLayout(subWindow_html_view)
        self.gridLayout.setObjectName(u"gridLayout")
        self.webEngineView = QWebEngineView(subWindow_html_view)
        self.webEngineView.setObjectName(u"webEngineView")

        self.gridLayout.addWidget(self.webEngineView, 0, 0, 1, 4)

        self.pushButton_save_data = QPushButton(subWindow_html_view)
        self.pushButton_save_data.setObjectName(u"pushButton_save_data")

        self.gridLayout.addWidget(self.pushButton_save_data, 1, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(311, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 0, 1, 1)

        self.pushButton_save_plot = QPushButton(subWindow_html_view)
        self.pushButton_save_plot.setObjectName(u"pushButton_save_plot")

        self.gridLayout.addWidget(self.pushButton_save_plot, 1, 1, 1, 1)


        self.retranslateUi(subWindow_html_view)

        QMetaObject.connectSlotsByName(subWindow_html_view)
    # setupUi

    def retranslateUi(self, subWindow_html_view):
        subWindow_html_view.setWindowTitle(QCoreApplication.translate("subWindow_html_view", u"Plot", None))
        self.pushButton_save_data.setText(QCoreApplication.translate("subWindow_html_view", u"Save Data", None))
        self.pushButton_save_plot.setText(QCoreApplication.translate("subWindow_html_view", u"Save Plot", None))
    # retranslateUi

