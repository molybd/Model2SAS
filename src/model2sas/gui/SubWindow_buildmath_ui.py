# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SubWindow_buildmath.ui'
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
from PySide6.QtWidgets import (QApplication, QLabel, QSizePolicy, QVBoxLayout,
    QWidget)

class Ui_subWindow_build_math_model(object):
    def setupUi(self, subWindow_build_math_model):
        if not subWindow_build_math_model.objectName():
            subWindow_build_math_model.setObjectName(u"subWindow_build_math_model")
        subWindow_build_math_model.resize(463, 352)
        self.verticalLayout = QVBoxLayout(subWindow_build_math_model)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(subWindow_build_math_model)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)


        self.retranslateUi(subWindow_build_math_model)

        QMetaObject.connectSlotsByName(subWindow_build_math_model)
    # setupUi

    def retranslateUi(self, subWindow_build_math_model):
        subWindow_build_math_model.setWindowTitle(QCoreApplication.translate("subWindow_build_math_model", u"Build Math Model", None))
        self.label.setText(QCoreApplication.translate("subWindow_build_math_model", u"some thing here", None))
    # retranslateUi

