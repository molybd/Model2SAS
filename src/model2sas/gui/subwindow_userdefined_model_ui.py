# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'subwindow_userdefined_model.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QTableView, QTextEdit,
    QWidget)

class Ui_subwindow_user_defined_model(object):
    def setupUi(self, subwindow_user_defined_model):
        if not subwindow_user_defined_model.objectName():
            subwindow_user_defined_model.setObjectName(u"subwindow_user_defined_model")
        subwindow_user_defined_model.resize(347, 461)
        self.gridLayout = QGridLayout(subwindow_user_defined_model)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_6 = QLabel(subwindow_user_defined_model)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)

        self.lineEdit_name = QLineEdit(subwindow_user_defined_model)
        self.lineEdit_name.setObjectName(u"lineEdit_name")

        self.gridLayout.addWidget(self.lineEdit_name, 0, 1, 1, 3)

        self.pushButton_help = QPushButton(subwindow_user_defined_model)
        self.pushButton_help.setObjectName(u"pushButton_help")

        self.gridLayout.addWidget(self.pushButton_help, 0, 4, 1, 1)

        self.label = QLabel(subwindow_user_defined_model)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 2)

        self.comboBox_coord = QComboBox(subwindow_user_defined_model)
        self.comboBox_coord.addItem("")
        self.comboBox_coord.addItem("")
        self.comboBox_coord.addItem("")
        self.comboBox_coord.setObjectName(u"comboBox_coord")

        self.gridLayout.addWidget(self.comboBox_coord, 1, 2, 1, 1)

        self.label_2 = QLabel(subwindow_user_defined_model)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 2)

        self.pushButton_add_param = QPushButton(subwindow_user_defined_model)
        self.pushButton_add_param.setObjectName(u"pushButton_add_param")

        self.gridLayout.addWidget(self.pushButton_add_param, 2, 2, 1, 1)

        self.pushButton_delete_param = QPushButton(subwindow_user_defined_model)
        self.pushButton_delete_param.setObjectName(u"pushButton_delete_param")

        self.gridLayout.addWidget(self.pushButton_delete_param, 2, 3, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(78, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 2, 4, 1, 2)

        self.tableView_params = QTableView(subwindow_user_defined_model)
        self.tableView_params.setObjectName(u"tableView_params")
        self.tableView_params.setMinimumSize(QSize(0, 100))
        font = QFont()
        font.setFamilies([u"Consolas"])
        self.tableView_params.setFont(font)

        self.gridLayout.addWidget(self.tableView_params, 3, 0, 1, 6)

        self.label_3 = QLabel(subwindow_user_defined_model)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_bound_point_1 = QLineEdit(subwindow_user_defined_model)
        self.lineEdit_bound_point_1.setObjectName(u"lineEdit_bound_point_1")

        self.horizontalLayout.addWidget(self.lineEdit_bound_point_1)

        self.lineEdit_bound_point_2 = QLineEdit(subwindow_user_defined_model)
        self.lineEdit_bound_point_2.setObjectName(u"lineEdit_bound_point_2")

        self.horizontalLayout.addWidget(self.lineEdit_bound_point_2)

        self.lineEdit_bound_point_3 = QLineEdit(subwindow_user_defined_model)
        self.lineEdit_bound_point_3.setObjectName(u"lineEdit_bound_point_3")

        self.horizontalLayout.addWidget(self.lineEdit_bound_point_3)


        self.gridLayout.addLayout(self.horizontalLayout, 4, 2, 1, 3)

        self.horizontalSpacer = QSpacerItem(2, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 4, 5, 1, 1)

        self.label_4 = QLabel(subwindow_user_defined_model)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 3)

        self.label_5 = QLabel(subwindow_user_defined_model)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 7, 0, 1, 3)

        self.pushButton_generate = QPushButton(subwindow_user_defined_model)
        self.pushButton_generate.setObjectName(u"pushButton_generate")

        self.gridLayout.addWidget(self.pushButton_generate, 9, 4, 1, 2)

        self.pushButton = QPushButton(subwindow_user_defined_model)
        self.pushButton.setObjectName(u"pushButton")

        self.gridLayout.addWidget(self.pushButton, 9, 3, 1, 1)

        self.textEdit_shape_description = QTextEdit(subwindow_user_defined_model)
        self.textEdit_shape_description.setObjectName(u"textEdit_shape_description")
        self.textEdit_shape_description.setFont(font)

        self.gridLayout.addWidget(self.textEdit_shape_description, 6, 0, 1, 6)

        self.textEdit_sld_description = QTextEdit(subwindow_user_defined_model)
        self.textEdit_sld_description.setObjectName(u"textEdit_sld_description")
        self.textEdit_sld_description.setFont(font)

        self.gridLayout.addWidget(self.textEdit_sld_description, 8, 0, 1, 6)

        QWidget.setTabOrder(self.lineEdit_name, self.pushButton_help)
        QWidget.setTabOrder(self.pushButton_help, self.comboBox_coord)
        QWidget.setTabOrder(self.comboBox_coord, self.pushButton_add_param)
        QWidget.setTabOrder(self.pushButton_add_param, self.pushButton_delete_param)
        QWidget.setTabOrder(self.pushButton_delete_param, self.tableView_params)
        QWidget.setTabOrder(self.tableView_params, self.lineEdit_bound_point_1)
        QWidget.setTabOrder(self.lineEdit_bound_point_1, self.lineEdit_bound_point_2)
        QWidget.setTabOrder(self.lineEdit_bound_point_2, self.lineEdit_bound_point_3)
        QWidget.setTabOrder(self.lineEdit_bound_point_3, self.textEdit_shape_description)
        QWidget.setTabOrder(self.textEdit_shape_description, self.textEdit_sld_description)
        QWidget.setTabOrder(self.textEdit_sld_description, self.pushButton_generate)

        self.retranslateUi(subwindow_user_defined_model)
        self.pushButton_help.clicked.connect(subwindow_user_defined_model.help)
        self.pushButton_add_param.clicked.connect(subwindow_user_defined_model.add_param)
        self.pushButton_delete_param.clicked.connect(subwindow_user_defined_model.delete_param)
        self.pushButton_generate.clicked.connect(subwindow_user_defined_model.generate)
        self.pushButton.clicked.connect(subwindow_user_defined_model.save)

        QMetaObject.connectSlotsByName(subwindow_user_defined_model)
    # setupUi

    def retranslateUi(self, subwindow_user_defined_model):
        subwindow_user_defined_model.setWindowTitle(QCoreApplication.translate("subwindow_user_defined_model", u"User-defined Model", None))
        self.label_6.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Name", None))
        self.lineEdit_name.setText("")
        self.lineEdit_name.setPlaceholderText(QCoreApplication.translate("subwindow_user_defined_model", u"use only a-z,A-Z,0-9,underscore", None))
        self.pushButton_help.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Help", None))
        self.label.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Coordinates", None))
        self.comboBox_coord.setItemText(0, QCoreApplication.translate("subwindow_user_defined_model", u"Cartesian", None))
        self.comboBox_coord.setItemText(1, QCoreApplication.translate("subwindow_user_defined_model", u"Spherical", None))
        self.comboBox_coord.setItemText(2, QCoreApplication.translate("subwindow_user_defined_model", u"Cylindrical", None))

        self.label_2.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Parameters", None))
        self.pushButton_add_param.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Add", None))
        self.pushButton_delete_param.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Delete", None))
        self.label_3.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Bound Point", None))
        self.label_4.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Shape Description", None))
        self.label_5.setText(QCoreApplication.translate("subwindow_user_defined_model", u"SLD Description", None))
        self.pushButton_generate.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Generate", None))
        self.pushButton.setText(QCoreApplication.translate("subwindow_user_defined_model", u"Save", None))
    # retranslateUi

