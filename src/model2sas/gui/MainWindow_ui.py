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
    QMainWindow, QMdiArea, QMenuBar, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QStatusBar, QTabWidget, QTableView, QTextBrowser,
    QTreeView, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1012, 690)
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
        self.menubar.setGeometry(QRect(0, 0, 1012, 22))
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
        self.label_length_unit = QLabel(self.dockWidgetContents_3)
        self.label_length_unit.setObjectName(u"label_length_unit")

        self.gridLayout_9.addWidget(self.label_length_unit, 0, 0, 1, 1)

        self.label_active_model = QLabel(self.dockWidgetContents_3)
        self.label_active_model.setObjectName(u"label_active_model")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_active_model.sizePolicy().hasHeightForWidth())
        self.label_active_model.setSizePolicy(sizePolicy)

        self.gridLayout_9.addWidget(self.label_active_model, 0, 2, 1, 1)

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


        self.gridLayout_9.addWidget(self.groupBox_part, 1, 0, 1, 2)

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
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.tableView_model_params.sizePolicy().hasHeightForWidth())
        self.tableView_model_params.setSizePolicy(sizePolicy2)
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
        self.pushButton_sample.setEnabled(True)

        self.gridLayout.addWidget(self.pushButton_sample, 1, 3, 1, 1)

        self.groupBox_transform = QGroupBox(self.tabPage_model)
        self.groupBox_transform.setObjectName(u"groupBox_transform")
        self.groupBox_transform.setEnabled(True)
        self.gridLayout_4 = QGridLayout(self.groupBox_transform)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.tableView_transforms = QTableView(self.groupBox_transform)
        self.tableView_transforms.setObjectName(u"tableView_transforms")
        self.tableView_transforms.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.tableView_transforms.sizePolicy().hasHeightForWidth())
        self.tableView_transforms.setSizePolicy(sizePolicy2)

        self.gridLayout_4.addWidget(self.tableView_transforms, 3, 0, 1, 4)

        self.pushButton_delete_selected_transform = QPushButton(self.groupBox_transform)
        self.pushButton_delete_selected_transform.setObjectName(u"pushButton_delete_selected_transform")
        self.pushButton_delete_selected_transform.setEnabled(False)

        self.gridLayout_4.addWidget(self.pushButton_delete_selected_transform, 0, 1, 1, 1)

        self.pushButton_add_transform = QPushButton(self.groupBox_transform)
        self.pushButton_add_transform.setObjectName(u"pushButton_add_transform")
        self.pushButton_add_transform.setEnabled(False)

        self.gridLayout_4.addWidget(self.pushButton_add_transform, 0, 0, 1, 1)

        self.pushButton_apply_transform = QPushButton(self.groupBox_transform)
        self.pushButton_apply_transform.setObjectName(u"pushButton_apply_transform")
        self.pushButton_apply_transform.setEnabled(False)

        self.gridLayout_4.addWidget(self.pushButton_apply_transform, 0, 2, 1, 2)


        self.gridLayout.addWidget(self.groupBox_transform, 2, 0, 1, 4)

        self.pushButton_plot_model = QPushButton(self.tabPage_model)
        self.pushButton_plot_model.setObjectName(u"pushButton_plot_model")
        self.pushButton_plot_model.setEnabled(True)

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
        self.gridLayout_8 = QGridLayout(self.tabPage_scattering)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.label_reciprocal_lattice_1d_size = QLabel(self.tabPage_scattering)
        self.label_reciprocal_lattice_1d_size.setObjectName(u"label_reciprocal_lattice_1d_size")

        self.gridLayout_8.addWidget(self.label_reciprocal_lattice_1d_size, 0, 0, 1, 2)

        self.lineEdit_reciprocal_lattice_1d_size = QLineEdit(self.tabPage_scattering)
        self.lineEdit_reciprocal_lattice_1d_size.setObjectName(u"lineEdit_reciprocal_lattice_1d_size")
        self.lineEdit_reciprocal_lattice_1d_size.setEnabled(False)

        self.gridLayout_8.addWidget(self.lineEdit_reciprocal_lattice_1d_size, 0, 2, 1, 1)

        self.groupBox_1d_measure = QGroupBox(self.tabPage_scattering)
        self.groupBox_1d_measure.setObjectName(u"groupBox_1d_measure")
        self.groupBox_1d_measure.setCheckable(False)
        self.gridLayout_10 = QGridLayout(self.groupBox_1d_measure)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
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


        self.gridLayout_10.addLayout(self.horizontalLayout, 0, 0, 1, 2)

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


        self.gridLayout_10.addLayout(self.horizontalLayout_2, 1, 0, 1, 2)

        self.horizontalSpacer_4 = QSpacerItem(190, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_10.addItem(self.horizontalSpacer_4, 2, 0, 1, 1)

        self.pushButton_measure_1d = QPushButton(self.groupBox_1d_measure)
        self.pushButton_measure_1d.setObjectName(u"pushButton_measure_1d")
        self.pushButton_measure_1d.setEnabled(True)

        self.gridLayout_10.addWidget(self.pushButton_measure_1d, 2, 1, 1, 1)


        self.gridLayout_8.addWidget(self.groupBox_1d_measure, 2, 0, 1, 3)

        self.groupBox_2d_measure = QGroupBox(self.tabPage_scattering)
        self.groupBox_2d_measure.setObjectName(u"groupBox_2d_measure")
        self.groupBox_2d_measure.setEnabled(True)
        self.gridLayout_3 = QGridLayout(self.groupBox_2d_measure)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.label_q2d_resolution = QLabel(self.groupBox_2d_measure)
        self.label_q2d_resolution.setObjectName(u"label_q2d_resolution")

        self.horizontalLayout_3.addWidget(self.label_q2d_resolution)

        self.lineEdit_det_res_h = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_det_res_h.setObjectName(u"lineEdit_det_res_h")

        self.horizontalLayout_3.addWidget(self.lineEdit_det_res_h)

        self.label_times = QLabel(self.groupBox_2d_measure)
        self.label_times.setObjectName(u"label_times")

        self.horizontalLayout_3.addWidget(self.label_times)

        self.lineEdit_det_res_v = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_det_res_v.setObjectName(u"lineEdit_det_res_v")

        self.horizontalLayout_3.addWidget(self.lineEdit_det_res_v)


        self.gridLayout_3.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)

        self.label_q2d_spacing = QLabel(self.groupBox_2d_measure)
        self.label_q2d_spacing.setObjectName(u"label_q2d_spacing")

        self.horizontalLayout_4.addWidget(self.label_q2d_spacing)

        self.lineEdit_det_pixel_size = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_det_pixel_size.setObjectName(u"lineEdit_det_pixel_size")
        self.lineEdit_det_pixel_size.setEnabled(True)

        self.horizontalLayout_4.addWidget(self.lineEdit_det_pixel_size)

        self.label_q2d_unit = QLabel(self.groupBox_2d_measure)
        self.label_q2d_unit.setObjectName(u"label_q2d_unit")

        self.horizontalLayout_4.addWidget(self.label_q2d_unit)


        self.gridLayout_3.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_5 = QLabel(self.groupBox_2d_measure)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_6.addWidget(self.label_5)

        self.lineEdit_wavelength = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_wavelength.setObjectName(u"lineEdit_wavelength")

        self.horizontalLayout_6.addWidget(self.lineEdit_wavelength)

        self.label_6 = QLabel(self.groupBox_2d_measure)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_6.addWidget(self.label_6)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer)

        self.label_3 = QLabel(self.groupBox_2d_measure)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_6.addWidget(self.label_3)

        self.lineEdit_det_sdd = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_det_sdd.setObjectName(u"lineEdit_det_sdd")
        self.lineEdit_det_sdd.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_det_sdd)

        self.label_4 = QLabel(self.groupBox_2d_measure)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_6.addWidget(self.label_4)


        self.gridLayout_3.addLayout(self.horizontalLayout_6, 2, 0, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_5)

        self.checkBox_log_Idet = QCheckBox(self.groupBox_2d_measure)
        self.checkBox_log_Idet.setObjectName(u"checkBox_log_Idet")

        self.horizontalLayout_5.addWidget(self.checkBox_log_Idet)

        self.pushButton_measure_det = QPushButton(self.groupBox_2d_measure)
        self.pushButton_measure_det.setObjectName(u"pushButton_measure_det")
        self.pushButton_measure_det.setEnabled(True)

        self.horizontalLayout_5.addWidget(self.pushButton_measure_det)


        self.gridLayout_3.addLayout(self.horizontalLayout_5, 3, 0, 1, 1)


        self.gridLayout_8.addWidget(self.groupBox_2d_measure, 3, 0, 1, 3)

        self.groupBox = QGroupBox(self.tabPage_scattering)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_2.addWidget(self.label_2, 0, 2, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.pushButton_measure_3d = QPushButton(self.groupBox)
        self.pushButton_measure_3d.setObjectName(u"pushButton_measure_3d")
        self.pushButton_measure_3d.setEnabled(True)

        self.gridLayout_2.addWidget(self.pushButton_measure_3d, 0, 4, 1, 1)

        self.lineEdit_q3d_qmax = QLineEdit(self.groupBox)
        self.lineEdit_q3d_qmax.setObjectName(u"lineEdit_q3d_qmax")

        self.gridLayout_2.addWidget(self.lineEdit_q3d_qmax, 0, 1, 1, 1)

        self.checkBox_log_I3d = QCheckBox(self.groupBox)
        self.checkBox_log_I3d.setObjectName(u"checkBox_log_I3d")

        self.gridLayout_2.addWidget(self.checkBox_log_I3d, 0, 3, 1, 1)


        self.gridLayout_8.addWidget(self.groupBox, 4, 0, 1, 3)

        self.verticalSpacer = QSpacerItem(20, 2, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_8.addItem(self.verticalSpacer, 5, 1, 1, 1)

        self.pushButton_scatter = QPushButton(self.tabPage_scattering)
        self.pushButton_scatter.setObjectName(u"pushButton_scatter")
        self.pushButton_scatter.setEnabled(True)

        self.gridLayout_8.addWidget(self.pushButton_scatter, 1, 2, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_8.addItem(self.horizontalSpacer_6, 1, 0, 1, 2)

        self.tabWidget_settings.addTab(self.tabPage_scattering, "")

        self.gridLayout_9.addWidget(self.tabWidget_settings, 1, 2, 3, 2)

        self.pushButton_add_to_assembly = QPushButton(self.dockWidgetContents_3)
        self.pushButton_add_to_assembly.setObjectName(u"pushButton_add_to_assembly")
        self.pushButton_add_to_assembly.setEnabled(False)

        self.gridLayout_9.addWidget(self.pushButton_add_to_assembly, 2, 0, 1, 1)

        self.comboBox_assemblies = QComboBox(self.dockWidgetContents_3)
        self.comboBox_assemblies.setObjectName(u"comboBox_assemblies")

        self.gridLayout_9.addWidget(self.comboBox_assemblies, 2, 1, 1, 1)

        self.groupBox_assembly = QGroupBox(self.dockWidgetContents_3)
        self.groupBox_assembly.setObjectName(u"groupBox_assembly")
        self.gridLayout_6 = QGridLayout(self.groupBox_assembly)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.treeView_assemblies = QTreeView(self.groupBox_assembly)
        self.treeView_assemblies.setObjectName(u"treeView_assemblies")
        sizePolicy2.setHeightForWidth(self.treeView_assemblies.sizePolicy().hasHeightForWidth())
        self.treeView_assemblies.setSizePolicy(sizePolicy2)
        self.treeView_assemblies.setMinimumSize(QSize(100, 0))
        self.treeView_assemblies.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.gridLayout_6.addWidget(self.treeView_assemblies, 3, 0, 1, 3)

        self.pushButton_new_assembly = QPushButton(self.groupBox_assembly)
        self.pushButton_new_assembly.setObjectName(u"pushButton_new_assembly")
        self.pushButton_new_assembly.setEnabled(True)

        self.gridLayout_6.addWidget(self.pushButton_new_assembly, 0, 0, 1, 3)


        self.gridLayout_9.addWidget(self.groupBox_assembly, 3, 0, 1, 2)

        self.horizontalSpacer_8 = QSpacerItem(101, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer_8, 4, 0, 1, 1)

        self.pushButton_delete_selected_model = QPushButton(self.dockWidgetContents_3)
        self.pushButton_delete_selected_model.setObjectName(u"pushButton_delete_selected_model")

        self.gridLayout_9.addWidget(self.pushButton_delete_selected_model, 4, 1, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(153, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer_7, 4, 2, 1, 1)

        self.progressBar = QProgressBar(self.dockWidgetContents_3)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setInvertedAppearance(False)

        self.gridLayout_9.addWidget(self.progressBar, 4, 3, 1, 1)

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
        QWidget.setTabOrder(self.pushButton_part_from_files, self.tableView_transforms)
        QWidget.setTabOrder(self.tableView_transforms, self.lineEdit_real_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_real_lattice_1d_size, self.pushButton_plot_model)
        QWidget.setTabOrder(self.pushButton_plot_model, self.tableView_model_params)
        QWidget.setTabOrder(self.tableView_model_params, self.pushButton_sample)
        QWidget.setTabOrder(self.pushButton_sample, self.lineEdit_reciprocal_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_reciprocal_lattice_1d_size, self.lineEdit_q1d_min)
        QWidget.setTabOrder(self.lineEdit_q1d_min, self.lineEdit_q1d_max)
        QWidget.setTabOrder(self.lineEdit_q1d_max, self.lineEdit_q1d_num)
        QWidget.setTabOrder(self.lineEdit_q1d_num, self.checkBox_q1d_log_spaced)
        QWidget.setTabOrder(self.checkBox_q1d_log_spaced, self.pushButton_measure_1d)
        QWidget.setTabOrder(self.pushButton_measure_1d, self.lineEdit_det_res_h)
        QWidget.setTabOrder(self.lineEdit_det_res_h, self.lineEdit_det_res_v)
        QWidget.setTabOrder(self.lineEdit_det_res_v, self.pushButton_measure_det)
        QWidget.setTabOrder(self.pushButton_measure_det, self.pushButton_new_assembly)

        self.retranslateUi(MainWindow)
        self.pushButton_part_from_files.clicked.connect(MainWindow.import_parts)
        self.treeView_parts.clicked.connect(MainWindow.part_model_selected)
        self.treeView_assemblies.clicked.connect(MainWindow.assembly_model_selected)
        self.pushButton_sample.clicked.connect(MainWindow.sample)
        self.pushButton_scatter.clicked.connect(MainWindow.scatter)
        self.pushButton_new_assembly.clicked.connect(MainWindow.new_assembly)
        self.pushButton_add_to_assembly.clicked.connect(MainWindow.add_to_assembly)
        self.pushButton_delete_selected_model.clicked.connect(MainWindow.delete_selected_model)
        self.pushButton_add_transform.clicked.connect(MainWindow.add_transform)
        self.pushButton_delete_selected_transform.clicked.connect(MainWindow.delete_selected_transform)
        self.pushButton_plot_model.clicked.connect(MainWindow.plot_model)
        self.pushButton_apply_transform.clicked.connect(MainWindow.apply_transform)
        self.pushButton_measure_1d.clicked.connect(MainWindow.measure_1d)
        self.pushButton_measure_det.clicked.connect(MainWindow.measure_det)
        self.pushButton_measure_3d.clicked.connect(MainWindow.measure_3d)

        self.tabWidget_settings.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Model2SAS", None))
        self.dockWidget_main.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main", None))
        self.label_length_unit.setText(QCoreApplication.translate("MainWindow", u"Length Unit: \u00c5", None))
        self.label_active_model.setText(QCoreApplication.translate("MainWindow", u"No Active Model", None))
        self.groupBox_part.setTitle(QCoreApplication.translate("MainWindow", u"Part Model", None))
        self.pushButton_part_from_files.setText(QCoreApplication.translate("MainWindow", u"Import From File(s)", None))
        self.groupBox_model_params.setTitle(QCoreApplication.translate("MainWindow", u"Model Parameters", None))
        self.label_real_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Points in longest edge", None))
        self.lineEdit_real_lattice_1d_size.setText("")
        self.pushButton_sample.setText(QCoreApplication.translate("MainWindow", u"Sample", None))
        self.groupBox_transform.setTitle(QCoreApplication.translate("MainWindow", u"Transform", None))
        self.pushButton_delete_selected_transform.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.pushButton_add_transform.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.pushButton_apply_transform.setText(QCoreApplication.translate("MainWindow", u"Apply Transform", None))
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
        self.pushButton_measure_1d.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.groupBox_2d_measure.setTitle(QCoreApplication.translate("MainWindow", u"2D Detector Simulation", None))
        self.label_q2d_resolution.setText(QCoreApplication.translate("MainWindow", u"Resolution ", None))
        self.label_times.setText(QCoreApplication.translate("MainWindow", u"\u00d7", None))
        self.label_q2d_spacing.setText(QCoreApplication.translate("MainWindow", u"Pixel size", None))
        self.label_q2d_unit.setText(QCoreApplication.translate("MainWindow", u"\u03bcm", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Wavelength", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u00c5", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"SDD", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_log_Idet.setText(QCoreApplication.translate("MainWindow", u"log(I)", None))
        self.pushButton_measure_det.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"3D", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u00c5^-1", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Q max", None))
        self.pushButton_measure_3d.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.checkBox_log_I3d.setText(QCoreApplication.translate("MainWindow", u"log(I)", None))
        self.pushButton_scatter.setText(QCoreApplication.translate("MainWindow", u"Virtual Scatter", None))
        self.tabWidget_settings.setTabText(self.tabWidget_settings.indexOf(self.tabPage_scattering), QCoreApplication.translate("MainWindow", u"Scattering", None))
        self.pushButton_add_to_assembly.setText(QCoreApplication.translate("MainWindow", u"Add to assembly", None))
        self.groupBox_assembly.setTitle(QCoreApplication.translate("MainWindow", u"Assembly Model", None))
        self.pushButton_new_assembly.setText(QCoreApplication.translate("MainWindow", u"New Assembly", None))
        self.pushButton_delete_selected_model.setText(QCoreApplication.translate("MainWindow", u"Delete Selected Model", None))
        self.dockWidget_log.setWindowTitle(QCoreApplication.translate("MainWindow", u"Log", None))
    # retranslateUi

