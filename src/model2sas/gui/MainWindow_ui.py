# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox,
    QComboBox, QDockWidget, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListView, QMainWindow, QMdiArea, QMenuBar,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QStatusBar, QTabWidget, QTableView, QTextBrowser,
    QTreeView, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1023, 661)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_7 = QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.mdiArea = QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName(u"mdiArea")
        self.mdiArea.setMinimumSize(QSize(400, 400))

        self.gridLayout_7.addWidget(self.mdiArea, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1023, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget_main = QDockWidget(MainWindow)
        self.dockWidget_main.setObjectName(u"dockWidget_main")
        self.dockWidgetContents_3 = QWidget()
        self.dockWidgetContents_3.setObjectName(u"dockWidgetContents_3")
        self.gridLayout_9 = QGridLayout(self.dockWidgetContents_3)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.groupBox_assembly = QGroupBox(self.dockWidgetContents_3)
        self.groupBox_assembly.setObjectName(u"groupBox_assembly")
        self.gridLayout_6 = QGridLayout(self.groupBox_assembly)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.treeView_assemblies = QTreeView(self.groupBox_assembly)
        self.treeView_assemblies.setObjectName(u"treeView_assemblies")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeView_assemblies.sizePolicy().hasHeightForWidth())
        self.treeView_assemblies.setSizePolicy(sizePolicy)
        self.treeView_assemblies.setMinimumSize(QSize(100, 0))
        self.treeView_assemblies.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.gridLayout_6.addWidget(self.treeView_assemblies, 3, 0, 1, 3)

        self.pushButton_new_assembly = QPushButton(self.groupBox_assembly)
        self.pushButton_new_assembly.setObjectName(u"pushButton_new_assembly")
        self.pushButton_new_assembly.setEnabled(True)

        self.gridLayout_6.addWidget(self.pushButton_new_assembly, 0, 0, 1, 3)


        self.gridLayout_9.addWidget(self.groupBox_assembly, 3, 0, 1, 3)

        self.label_length_unit = QLabel(self.dockWidgetContents_3)
        self.label_length_unit.setObjectName(u"label_length_unit")

        self.gridLayout_9.addWidget(self.label_length_unit, 0, 0, 1, 1)

        self.pushButton_delete_selected_model = QPushButton(self.dockWidgetContents_3)
        self.pushButton_delete_selected_model.setObjectName(u"pushButton_delete_selected_model")

        self.gridLayout_9.addWidget(self.pushButton_delete_selected_model, 4, 2, 1, 1)

        self.tabWidget_settings = QTabWidget(self.dockWidgetContents_3)
        self.tabWidget_settings.setObjectName(u"tabWidget_settings")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabWidget_settings.sizePolicy().hasHeightForWidth())
        self.tabWidget_settings.setSizePolicy(sizePolicy1)
        self.tabPage_model = QWidget()
        self.tabPage_model.setObjectName(u"tabPage_model")
        self.gridLayout = QGridLayout(self.tabPage_model)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox_model_params = QGroupBox(self.tabPage_model)
        self.groupBox_model_params.setObjectName(u"groupBox_model_params")
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_model_params)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.tableView_model_params = QTableView(self.groupBox_model_params)
        self.tableView_model_params.setObjectName(u"tableView_model_params")
        self.tableView_model_params.setEnabled(False)
        sizePolicy.setHeightForWidth(self.tableView_model_params.sizePolicy().hasHeightForWidth())
        self.tableView_model_params.setSizePolicy(sizePolicy)
        self.tableView_model_params.setAlternatingRowColors(True)

        self.verticalLayout_5.addWidget(self.tableView_model_params)


        self.gridLayout.addWidget(self.groupBox_model_params, 0, 0, 1, 4)

        self.label_real_lattice_1d_size = QLabel(self.tabPage_model)
        self.label_real_lattice_1d_size.setObjectName(u"label_real_lattice_1d_size")

        self.gridLayout.addWidget(self.label_real_lattice_1d_size, 1, 0, 1, 2)

        self.lineEdit_real_lattice_1d_size = QLineEdit(self.tabPage_model)
        self.lineEdit_real_lattice_1d_size.setObjectName(u"lineEdit_real_lattice_1d_size")

        self.gridLayout.addWidget(self.lineEdit_real_lattice_1d_size, 1, 2, 1, 1)

        self.pushButton_sample = QPushButton(self.tabPage_model)
        self.pushButton_sample.setObjectName(u"pushButton_sample")
        self.pushButton_sample.setEnabled(False)

        self.gridLayout.addWidget(self.pushButton_sample, 1, 3, 1, 1)

        self.groupBox_transform = QGroupBox(self.tabPage_model)
        self.groupBox_transform.setObjectName(u"groupBox_transform")
        self.groupBox_transform.setEnabled(True)
        self.gridLayout_4 = QGridLayout(self.groupBox_transform)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.pushButton_delete_selected_transform = QPushButton(self.groupBox_transform)
        self.pushButton_delete_selected_transform.setObjectName(u"pushButton_delete_selected_transform")
        self.pushButton_delete_selected_transform.setEnabled(False)

        self.gridLayout_4.addWidget(self.pushButton_delete_selected_transform, 7, 3, 1, 1)

        self.label_transform_vector = QLabel(self.groupBox_transform)
        self.label_transform_vector.setObjectName(u"label_transform_vector")
        self.label_transform_vector.setEnabled(True)

        self.gridLayout_4.addWidget(self.label_transform_vector, 1, 0, 1, 1)

        self.listView_transforms = QListView(self.groupBox_transform)
        self.listView_transforms.setObjectName(u"listView_transforms")
        self.listView_transforms.setEnabled(True)
        sizePolicy.setHeightForWidth(self.listView_transforms.sizePolicy().hasHeightForWidth())
        self.listView_transforms.setSizePolicy(sizePolicy)

        self.gridLayout_4.addWidget(self.listView_transforms, 6, 0, 1, 4)

        self.label_transform_angle = QLabel(self.groupBox_transform)
        self.label_transform_angle.setObjectName(u"label_transform_angle")
        self.label_transform_angle.setEnabled(True)

        self.gridLayout_4.addWidget(self.label_transform_angle, 2, 0, 1, 1)

        self.lineEdit_transform_angle = QLineEdit(self.groupBox_transform)
        self.lineEdit_transform_angle.setObjectName(u"lineEdit_transform_angle")
        self.lineEdit_transform_angle.setEnabled(True)

        self.gridLayout_4.addWidget(self.lineEdit_transform_angle, 2, 1, 1, 1)

        self.pushButton_add_transform = QPushButton(self.groupBox_transform)
        self.pushButton_add_transform.setObjectName(u"pushButton_add_transform")
        self.pushButton_add_transform.setEnabled(False)

        self.gridLayout_4.addWidget(self.pushButton_add_transform, 2, 3, 1, 1)

        self.comboBox_transform_type = QComboBox(self.groupBox_transform)
        self.comboBox_transform_type.addItem("")
        self.comboBox_transform_type.addItem("")
        self.comboBox_transform_type.setObjectName(u"comboBox_transform_type")
        self.comboBox_transform_type.setEnabled(True)

        self.gridLayout_4.addWidget(self.comboBox_transform_type, 0, 0, 1, 4)

        self.lineEdit_transform_vector = QLineEdit(self.groupBox_transform)
        self.lineEdit_transform_vector.setObjectName(u"lineEdit_transform_vector")
        self.lineEdit_transform_vector.setEnabled(True)

        self.gridLayout_4.addWidget(self.lineEdit_transform_vector, 1, 1, 1, 3)

        self.label_degree = QLabel(self.groupBox_transform)
        self.label_degree.setObjectName(u"label_degree")

        self.gridLayout_4.addWidget(self.label_degree, 2, 2, 1, 1)


        self.gridLayout.addWidget(self.groupBox_transform, 2, 0, 1, 4)

        self.pushButton_plot_model = QPushButton(self.tabPage_model)
        self.pushButton_plot_model.setObjectName(u"pushButton_plot_model")
        self.pushButton_plot_model.setEnabled(False)

        self.gridLayout.addWidget(self.pushButton_plot_model, 3, 2, 1, 1)

        self.radioButton_voxel_plot = QRadioButton(self.tabPage_model)
        self.buttonGroup = QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.radioButton_voxel_plot)
        self.radioButton_voxel_plot.setObjectName(u"radioButton_voxel_plot")
        self.radioButton_voxel_plot.setEnabled(True)
        self.radioButton_voxel_plot.setChecked(True)

        self.gridLayout.addWidget(self.radioButton_voxel_plot, 3, 0, 1, 1)

        self.radioButton_volume_plot = QRadioButton(self.tabPage_model)
        self.buttonGroup.addButton(self.radioButton_volume_plot)
        self.radioButton_volume_plot.setObjectName(u"radioButton_volume_plot")
        self.radioButton_volume_plot.setEnabled(True)

        self.gridLayout.addWidget(self.radioButton_volume_plot, 3, 1, 1, 1)

        self.tabWidget_settings.addTab(self.tabPage_model, "")
        self.tabPage_scattering = QWidget()
        self.tabPage_scattering.setObjectName(u"tabPage_scattering")
        self.gridLayout_3 = QGridLayout(self.tabPage_scattering)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_reciprocal_lattice_1d_size = QLabel(self.tabPage_scattering)
        self.label_reciprocal_lattice_1d_size.setObjectName(u"label_reciprocal_lattice_1d_size")

        self.gridLayout_3.addWidget(self.label_reciprocal_lattice_1d_size, 0, 0, 1, 1)

        self.lineEdit_reciprocal_lattice_1d_size = QLineEdit(self.tabPage_scattering)
        self.lineEdit_reciprocal_lattice_1d_size.setObjectName(u"lineEdit_reciprocal_lattice_1d_size")
        self.lineEdit_reciprocal_lattice_1d_size.setEnabled(False)

        self.gridLayout_3.addWidget(self.lineEdit_reciprocal_lattice_1d_size, 0, 1, 1, 1)

        self.tabWidget_measure = QTabWidget(self.tabPage_scattering)
        self.tabWidget_measure.setObjectName(u"tabWidget_measure")
        self.tabWidget_measure.setEnabled(True)
        self.tabPage_direct_measure = QWidget()
        self.tabPage_direct_measure.setObjectName(u"tabPage_direct_measure")
        self.verticalLayout_3 = QVBoxLayout(self.tabPage_direct_measure)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.groupBox_1d_measure = QGroupBox(self.tabPage_direct_measure)
        self.groupBox_1d_measure.setObjectName(u"groupBox_1d_measure")
        self.groupBox_1d_measure.setCheckable(False)
        self.verticalLayout = QVBoxLayout(self.groupBox_1d_measure)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_q1d_range = QLabel(self.groupBox_1d_measure)
        self.label_q1d_range.setObjectName(u"label_q1d_range")

        self.horizontalLayout.addWidget(self.label_q1d_range)

        self.lineEdit_q1d_min = QLineEdit(self.groupBox_1d_measure)
        self.lineEdit_q1d_min.setObjectName(u"lineEdit_q1d_min")

        self.horizontalLayout.addWidget(self.lineEdit_q1d_min)

        self.label_dash = QLabel(self.groupBox_1d_measure)
        self.label_dash.setObjectName(u"label_dash")

        self.horizontalLayout.addWidget(self.label_dash)

        self.lineEdit_q1d_max = QLineEdit(self.groupBox_1d_measure)
        self.lineEdit_q1d_max.setObjectName(u"lineEdit_q1d_max")

        self.horizontalLayout.addWidget(self.lineEdit_q1d_max)

        self.label_q1d_unit = QLabel(self.groupBox_1d_measure)
        self.label_q1d_unit.setObjectName(u"label_q1d_unit")

        self.horizontalLayout.addWidget(self.label_q1d_unit)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_q1d_num = QLabel(self.groupBox_1d_measure)
        self.label_q1d_num.setObjectName(u"label_q1d_num")

        self.horizontalLayout_2.addWidget(self.label_q1d_num)

        self.lineEdit_q1d_num = QLineEdit(self.groupBox_1d_measure)
        self.lineEdit_q1d_num.setObjectName(u"lineEdit_q1d_num")

        self.horizontalLayout_2.addWidget(self.lineEdit_q1d_num)

        self.checkBox_q1d_log_spaced = QCheckBox(self.groupBox_1d_measure)
        self.checkBox_q1d_log_spaced.setObjectName(u"checkBox_q1d_log_spaced")

        self.horizontalLayout_2.addWidget(self.checkBox_q1d_log_spaced)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.pushButton_1d_measure = QPushButton(self.groupBox_1d_measure)
        self.pushButton_1d_measure.setObjectName(u"pushButton_1d_measure")
        self.pushButton_1d_measure.setEnabled(False)

        self.verticalLayout.addWidget(self.pushButton_1d_measure)


        self.verticalLayout_3.addWidget(self.groupBox_1d_measure)

        self.groupBox_2d_measure = QGroupBox(self.tabPage_direct_measure)
        self.groupBox_2d_measure.setObjectName(u"groupBox_2d_measure")
        self.groupBox_2d_measure.setEnabled(False)
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_2d_measure)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_q2d_resolution = QLabel(self.groupBox_2d_measure)
        self.label_q2d_resolution.setObjectName(u"label_q2d_resolution")

        self.horizontalLayout_3.addWidget(self.label_q2d_resolution)

        self.lineEdit_q2d_resolution_h = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_resolution_h.setObjectName(u"lineEdit_q2d_resolution_h")

        self.horizontalLayout_3.addWidget(self.lineEdit_q2d_resolution_h)

        self.label_times = QLabel(self.groupBox_2d_measure)
        self.label_times.setObjectName(u"label_times")

        self.horizontalLayout_3.addWidget(self.label_times)

        self.lineEdit_q2d_resolution_v = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_resolution_v.setObjectName(u"lineEdit_q2d_resolution_v")

        self.horizontalLayout_3.addWidget(self.lineEdit_q2d_resolution_v)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_q2d_spacing = QLabel(self.groupBox_2d_measure)
        self.label_q2d_spacing.setObjectName(u"label_q2d_spacing")

        self.horizontalLayout_4.addWidget(self.label_q2d_spacing)

        self.lineEdit_q2d_spacing = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_spacing.setObjectName(u"lineEdit_q2d_spacing")

        self.horizontalLayout_4.addWidget(self.lineEdit_q2d_spacing)

        self.label_q2d_unit = QLabel(self.groupBox_2d_measure)
        self.label_q2d_unit.setObjectName(u"label_q2d_unit")

        self.horizontalLayout_4.addWidget(self.label_q2d_unit)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_q2d_normal_vector = QLabel(self.groupBox_2d_measure)
        self.label_q2d_normal_vector.setObjectName(u"label_q2d_normal_vector")

        self.horizontalLayout_6.addWidget(self.label_q2d_normal_vector)

        self.lineEdit_q2d_normal_x = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_normal_x.setObjectName(u"lineEdit_q2d_normal_x")
        self.lineEdit_q2d_normal_x.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_q2d_normal_x)

        self.lineEdit_q2d_normal_y = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_normal_y.setObjectName(u"lineEdit_q2d_normal_y")
        self.lineEdit_q2d_normal_y.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_q2d_normal_y)

        self.lineEdit_q2d_normal_z = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_normal_z.setObjectName(u"lineEdit_q2d_normal_z")
        self.lineEdit_q2d_normal_z.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_q2d_normal_z)


        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.checkBox_2d_show_3d = QCheckBox(self.groupBox_2d_measure)
        self.checkBox_2d_show_3d.setObjectName(u"checkBox_2d_show_3d")

        self.horizontalLayout_5.addWidget(self.checkBox_2d_show_3d)

        self.pushButton_2d_measure = QPushButton(self.groupBox_2d_measure)
        self.pushButton_2d_measure.setObjectName(u"pushButton_2d_measure")

        self.horizontalLayout_5.addWidget(self.pushButton_2d_measure)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)


        self.verticalLayout_3.addWidget(self.groupBox_2d_measure)

        self.verticalSpacer = QSpacerItem(228, 132, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.tabWidget_measure.addTab(self.tabPage_direct_measure, "")
        self.tabPage_virtual_detector = QWidget()
        self.tabPage_virtual_detector.setObjectName(u"tabPage_virtual_detector")
        self.tabWidget_measure.addTab(self.tabPage_virtual_detector, "")

        self.gridLayout_3.addWidget(self.tabWidget_measure, 2, 0, 1, 2)

        self.pushButton_scatter = QPushButton(self.tabPage_scattering)
        self.pushButton_scatter.setObjectName(u"pushButton_scatter")
        self.pushButton_scatter.setEnabled(False)

        self.gridLayout_3.addWidget(self.pushButton_scatter, 1, 0, 1, 2)

        self.tabWidget_settings.addTab(self.tabPage_scattering, "")

        self.gridLayout_9.addWidget(self.tabWidget_settings, 1, 3, 4, 1)

        self.label_active_model = QLabel(self.dockWidgetContents_3)
        self.label_active_model.setObjectName(u"label_active_model")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_active_model.sizePolicy().hasHeightForWidth())
        self.label_active_model.setSizePolicy(sizePolicy2)

        self.gridLayout_9.addWidget(self.label_active_model, 0, 3, 1, 1)

        self.groupBox_part = QGroupBox(self.dockWidgetContents_3)
        self.groupBox_part.setObjectName(u"groupBox_part")
        self.gridLayout_5 = QGridLayout(self.groupBox_part)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.treeView_parts = QTreeView(self.groupBox_part)
        self.treeView_parts.setObjectName(u"treeView_parts")

        self.gridLayout_5.addWidget(self.treeView_parts, 2, 0, 1, 3)

        self.pushButton_part_from_files = QPushButton(self.groupBox_part)
        self.pushButton_part_from_files.setObjectName(u"pushButton_part_from_files")

        self.gridLayout_5.addWidget(self.pushButton_part_from_files, 1, 0, 1, 2)


        self.gridLayout_9.addWidget(self.groupBox_part, 1, 0, 1, 3)

        self.pushButton_add_to_assembly = QPushButton(self.dockWidgetContents_3)
        self.pushButton_add_to_assembly.setObjectName(u"pushButton_add_to_assembly")
        self.pushButton_add_to_assembly.setEnabled(False)

        self.gridLayout_9.addWidget(self.pushButton_add_to_assembly, 2, 0, 1, 2)

        self.comboBox_assemblies = QComboBox(self.dockWidgetContents_3)
        self.comboBox_assemblies.setObjectName(u"comboBox_assemblies")

        self.gridLayout_9.addWidget(self.comboBox_assemblies, 2, 2, 1, 1)

        self.dockWidget_main.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_main)
        self.dockWidget_log = QDockWidget(MainWindow)
        self.dockWidget_log.setObjectName(u"dockWidget_log")
        self.dockWidgetContents_4 = QWidget()
        self.dockWidgetContents_4.setObjectName(u"dockWidgetContents_4")
        self.verticalLayout_4 = QVBoxLayout(self.dockWidgetContents_4)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.textBrowser_log = QTextBrowser(self.dockWidgetContents_4)
        self.textBrowser_log.setObjectName(u"textBrowser_log")
        self.textBrowser_log.setEnabled(True)
        font = QFont()
        font.setFamilies([u"Consolas"])
        self.textBrowser_log.setFont(font)

        self.verticalLayout_4.addWidget(self.textBrowser_log)

        self.dockWidget_log.setWidget(self.dockWidgetContents_4)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_log)
        QWidget.setTabOrder(self.pushButton_part_from_files, self.comboBox_transform_type)
        QWidget.setTabOrder(self.comboBox_transform_type, self.listView_transforms)
        QWidget.setTabOrder(self.listView_transforms, self.lineEdit_real_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_real_lattice_1d_size, self.pushButton_plot_model)
        QWidget.setTabOrder(self.pushButton_plot_model, self.tableView_model_params)
        QWidget.setTabOrder(self.tableView_model_params, self.pushButton_sample)
        QWidget.setTabOrder(self.pushButton_sample, self.lineEdit_reciprocal_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_reciprocal_lattice_1d_size, self.pushButton_scatter)
        QWidget.setTabOrder(self.pushButton_scatter, self.tabWidget_measure)
        QWidget.setTabOrder(self.tabWidget_measure, self.lineEdit_q1d_min)
        QWidget.setTabOrder(self.lineEdit_q1d_min, self.lineEdit_q1d_max)
        QWidget.setTabOrder(self.lineEdit_q1d_max, self.lineEdit_q1d_num)
        QWidget.setTabOrder(self.lineEdit_q1d_num, self.checkBox_q1d_log_spaced)
        QWidget.setTabOrder(self.checkBox_q1d_log_spaced, self.pushButton_1d_measure)
        QWidget.setTabOrder(self.pushButton_1d_measure, self.lineEdit_q2d_resolution_h)
        QWidget.setTabOrder(self.lineEdit_q2d_resolution_h, self.lineEdit_q2d_resolution_v)
        QWidget.setTabOrder(self.lineEdit_q2d_resolution_v, self.lineEdit_q2d_spacing)
        QWidget.setTabOrder(self.lineEdit_q2d_spacing, self.lineEdit_q2d_normal_x)
        QWidget.setTabOrder(self.lineEdit_q2d_normal_x, self.lineEdit_q2d_normal_y)
        QWidget.setTabOrder(self.lineEdit_q2d_normal_y, self.lineEdit_q2d_normal_z)
        QWidget.setTabOrder(self.lineEdit_q2d_normal_z, self.checkBox_2d_show_3d)
        QWidget.setTabOrder(self.checkBox_2d_show_3d, self.pushButton_2d_measure)
        QWidget.setTabOrder(self.pushButton_2d_measure, self.pushButton_new_assembly)

        self.retranslateUi(MainWindow)
        self.pushButton_part_from_files.clicked.connect(MainWindow.import_parts)
        self.treeView_parts.clicked.connect(MainWindow.part_model_selected)
        self.treeView_assemblies.clicked.connect(MainWindow.assembly_model_selected)
        self.pushButton_sample.clicked.connect(MainWindow.sample)
        self.pushButton_scatter.clicked.connect(MainWindow.scatter)
        self.pushButton_1d_measure.clicked.connect(MainWindow.measure)
        self.pushButton_new_assembly.clicked.connect(MainWindow.new_assembly)
        self.pushButton_add_to_assembly.clicked.connect(MainWindow.add_to_assembly)
        self.pushButton_delete_selected_model.clicked.connect(MainWindow.delete_selected_model)
        self.pushButton_add_transform.clicked.connect(MainWindow.add_transform)
        self.pushButton_delete_selected_transform.clicked.connect(MainWindow.delete_selected_transform)
        self.pushButton_plot_model.clicked.connect(MainWindow.plot_model)

        self.tabWidget_settings.setCurrentIndex(0)
        self.tabWidget_measure.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Model2SAS", None))
        self.dockWidget_main.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main", None))
        self.groupBox_assembly.setTitle(QCoreApplication.translate("MainWindow", u"Assembly Model", None))
        self.pushButton_new_assembly.setText(QCoreApplication.translate("MainWindow", u"New Assembly", None))
        self.label_length_unit.setText(QCoreApplication.translate("MainWindow", u"Length Unit: \u00c5", None))
        self.pushButton_delete_selected_model.setText(QCoreApplication.translate("MainWindow", u"Delete Selected Model", None))
        self.groupBox_model_params.setTitle(QCoreApplication.translate("MainWindow", u"Model Parameters", None))
        self.label_real_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Points in longest edge", None))
        self.lineEdit_real_lattice_1d_size.setText("")
        self.pushButton_sample.setText(QCoreApplication.translate("MainWindow", u"Sample", None))
        self.groupBox_transform.setTitle(QCoreApplication.translate("MainWindow", u"Transform", None))
        self.pushButton_delete_selected_transform.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.label_transform_vector.setText(QCoreApplication.translate("MainWindow", u"Vector/Axis", None))
        self.label_transform_angle.setText(QCoreApplication.translate("MainWindow", u"Angle", None))
        self.lineEdit_transform_angle.setText(QCoreApplication.translate("MainWindow", u"90", None))
        self.pushButton_add_transform.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.comboBox_transform_type.setItemText(0, QCoreApplication.translate("MainWindow", u"Translate", None))
        self.comboBox_transform_type.setItemText(1, QCoreApplication.translate("MainWindow", u"Rotate", None))

        self.lineEdit_transform_vector.setText(QCoreApplication.translate("MainWindow", u"0,0,1", None))
        self.label_degree.setText(QCoreApplication.translate("MainWindow", u"deg", None))
        self.pushButton_plot_model.setText(QCoreApplication.translate("MainWindow", u"Plot", None))
        self.radioButton_voxel_plot.setText(QCoreApplication.translate("MainWindow", u"Voxel", None))
        self.radioButton_volume_plot.setText(QCoreApplication.translate("MainWindow", u"Volume", None))
        self.tabWidget_settings.setTabText(self.tabWidget_settings.indexOf(self.tabPage_model), QCoreApplication.translate("MainWindow", u"Model Settings", None))
        self.label_reciprocal_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Edge points num in reciprocal grid", None))
        self.lineEdit_reciprocal_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"default", None))
        self.groupBox_1d_measure.setTitle(QCoreApplication.translate("MainWindow", u"1D", None))
        self.label_q1d_range.setText(QCoreApplication.translate("MainWindow", u"Q range", None))
        self.label_dash.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.label_q1d_unit.setText(QCoreApplication.translate("MainWindow", u"\u00c5^-1", None))
        self.label_q1d_num.setText(QCoreApplication.translate("MainWindow", u"Q num", None))
        self.checkBox_q1d_log_spaced.setText(QCoreApplication.translate("MainWindow", u"Log Spaced", None))
        self.pushButton_1d_measure.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.groupBox_2d_measure.setTitle(QCoreApplication.translate("MainWindow", u"2D", None))
        self.label_q2d_resolution.setText(QCoreApplication.translate("MainWindow", u"Resolution ", None))
        self.label_times.setText(QCoreApplication.translate("MainWindow", u"\u00d7", None))
        self.label_q2d_spacing.setText(QCoreApplication.translate("MainWindow", u"Q spacing", None))
        self.label_q2d_unit.setText(QCoreApplication.translate("MainWindow", u"\u00c5^-1", None))
        self.label_q2d_normal_vector.setText(QCoreApplication.translate("MainWindow", u"Normal vector", None))
        self.checkBox_2d_show_3d.setText(QCoreApplication.translate("MainWindow", u"Show in 3D Q Space", None))
        self.pushButton_2d_measure.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.tabWidget_measure.setTabText(self.tabWidget_measure.indexOf(self.tabPage_direct_measure), QCoreApplication.translate("MainWindow", u"Direct Measure", None))
        self.tabWidget_measure.setTabText(self.tabWidget_measure.indexOf(self.tabPage_virtual_detector), QCoreApplication.translate("MainWindow", u"Virtual Detector", None))
        self.pushButton_scatter.setText(QCoreApplication.translate("MainWindow", u"Virtual Scatter", None))
        self.tabWidget_settings.setTabText(self.tabWidget_settings.indexOf(self.tabPage_scattering), QCoreApplication.translate("MainWindow", u"Scattering Settings", None))
        self.label_active_model.setText(QCoreApplication.translate("MainWindow", u"No Active Model", None))
        self.groupBox_part.setTitle(QCoreApplication.translate("MainWindow", u"Part Model", None))
        self.pushButton_part_from_files.setText(QCoreApplication.translate("MainWindow", u"Import From File(s)", None))
        self.pushButton_add_to_assembly.setText(QCoreApplication.translate("MainWindow", u"Add to assembly", None))
        self.dockWidget_log.setWindowTitle(QCoreApplication.translate("MainWindow", u"Log", None))
    # retranslateUi

