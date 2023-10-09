# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SubWindow_htmlview.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
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
from PySide6.QtWidgets import (QApplication, QSizePolicy, QVBoxLayout, QWidget)

class Ui_subWindow_html_view(object):
    def setupUi(self, subWindow_html_view):
        if not subWindow_html_view.objectName():
            subWindow_html_view.setObjectName(u"subWindow_html_view")
        subWindow_html_view.resize(400, 300)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(subWindow_html_view.sizePolicy().hasHeightForWidth())
        subWindow_html_view.setSizePolicy(sizePolicy)
        subWindow_html_view.setMinimumSize(QSize(400, 300))
        self.verticalLayout = QVBoxLayout(subWindow_html_view)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.webEngineView = QWebEngineView(subWindow_html_view)
        self.webEngineView.setObjectName(u"webEngineView")

        self.verticalLayout.addWidget(self.webEngineView)


        self.retranslateUi(subWindow_html_view)

        QMetaObject.connectSlotsByName(subWindow_html_view)
    # setupUi

    def retranslateUi(self, subWindow_html_view):
        subWindow_html_view.setWindowTitle(QCoreApplication.translate("subWindow_html_view", u"Plot", None))
    # retranslateUi

