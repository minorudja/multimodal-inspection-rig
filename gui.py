import sys
import json
from PyQt6.QtWidgets import (QMainWindow, QWidget, QGridLayout, 
                             QLabel, QGroupBox, QVBoxLayout, QFrame, QTextEdit,
                             QStackedWidget, QSlider, QDoubleSpinBox, QSpinBox, 
                             QPushButton, QHBoxLayout, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

# Custom modules for Hardware, Camera Threading, and LLM Logic
from hardware import ActuatorController
from worker import RealSenseThread
from reasoner import LLMWorker

# =========================================================
# MAIN GUI - Multimodal Inspection Rig
# =========================================================
class VisionDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Inspection Rig Interface")
        self.resize(1600, 900)
        
        # --- HARDWARE INIT ---
        # Change 'COM4' here if the Arduino is assigned a different port
        self.actuator = ActuatorController(port='COM4')

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QGridLayout()
        self.central_widget.setLayout(self.grid_layout)

        # Dashboard layout weights (Column 0 is narrow for controls, 1 and 2 are wide for feeds)
        self.grid_layout.setColumnStretch(0, 1) 
        self.grid_layout.setColumnStretch(1, 3) 
        self.grid_layout.setColumnStretch(2, 3) 
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)

        # Initialize the 6 UI Tiles
        self.tile_1 = self.setup_tile_1()
        self.grid_layout.addWidget(self.tile_1, 0, 0)
        self.tile_2 = self.setup_tile_2()
        self.grid_layout.addWidget(self.tile_2, 1, 0)
        
        # Tile 3 handles RGB and the dynamically updated Legend
        self.tile_3_frame, self.tile_3_label, self.tile_3_legend_layout = self.setup_tile_3()
        self.grid_layout.addWidget(self.tile_3_frame, 0, 1)
        
        # Tile 4: Visualizing defect crops or detections
        self.tile_4_frame, self.tile_4_label = self.setup_tile_4()
        self.grid_layout.addWidget(self.tile_4_frame, 1, 1)
        
        # Tile 5: Uses a stack to switch between Depth Map and 3D PCD Render
        self.tile_5_stack = self.setup_tile_5()
        self.grid_layout.addWidget(self.tile_5_stack, 0, 2)
        
        # Tile 6: Dedicated log for the LLM reasoner output
        self.tile_6 = self.setup_tile_6()
        self.grid_layout.addWidget(self.tile_6, 1, 2)

        # --- THREADING & SIGNAL CONNECTIONS ---
        self.thread = RealSenseThread()
        # Feed updates
        self.thread.change_rgb_signal.connect(self.update_tile_3_rgb)
        self.thread.change_depth_signal.connect(self.update_tile_5_depth)
        self.thread.change_defect_signal.connect(self.update_tile_4_defect)
        self.thread.change_pcd_signal.connect(self.update_tile_5_pcd)
        # Stats and UI labels
        self.thread.update_legend_signal.connect(self.update_legend)
        self.thread.update_stats_signal.connect(self.update_material_stats)
        # Hardware/Logical triggers
        self.thread.request_motor_move.connect(self.handle_motor_request)
        self.thread.request_llm_analysis_signal.connect(self.run_reasoner_auto)
        self.thread.log_reasoning_signal.connect(self.update_reasoning_log)
        self.thread.system_log_signal.connect(self.update_log)
        
        # Reasoner worker (for non-blocking LLM calls)
        self.reasoner_thread = LLMWorker()
        self.reasoner_thread.finished_signal.connect(self.handle_reasoner_output)
        
        self.thread.start()

        self.lbl_plastic_stat = None
        self.lbl_copper_stat = None

        # Quick check if serial actually opened
        if self.actuator.ser and self.actuator.ser.is_open:
            self.update_log(f"SYSTEM: Arduino connected on {self.actuator.port}")
        else:
            self.update_log(f"ERROR: Could not connect to Arduino on {self.actuator.port}")

    def closeEvent(self, event):
        """Clean shutdown: stop the camera thread and close serial port."""
        self.thread.stop()
        self.actuator.close()
        event.accept()

    # --- HARDWARE SLOTS ---
    @pyqtSlot(int, float, float)
    def handle_motor_request(self, act_id, val, speed):
        """Bridge between camera logic and physical actuators."""
        if act_id == 1:
            self.actuator.rotate_turntable(val, speed)
        else:
            self.actuator.move_linear_actuator(act_id, val, speed)

    def on_toggle_scan(self):
        """Controls the 3D capture phase. Disables Inspection to avoid conflicts."""
        if self.btn_scan.isChecked():
            self.btn_scan.setText("STOP 3D SCAN")
            self.btn_scan.setStyleSheet("background-color: #e74c3c; font-weight: bold;")
            self.btn_inspect.setEnabled(False)
            self.btn_inspect.setStyleSheet("background-color: #555; color: #888; font-weight: bold;")
            
            self.thread.start_scan_sequence()
            self.tile_5_stack.setCurrentIndex(1) # Switch to PCD view
            self.thread.combined_pcd.clear()
        else:
            self.btn_scan.setText("Start 3D Scan")
            self.btn_scan.setStyleSheet("background-color: #27ae60; font-weight: bold;")
            
            self.thread.stop_scan_sequence()
            self.actuator.stop_actuator(2) 
            self.actuator.stop_actuator(1)

            self.btn_inspect.setEnabled(True)
            self.btn_inspect.setStyleSheet("background-color: #d35400; font-weight: bold;") 
            self.update_log("SYSTEM: Scan Phase Complete. Inspection Enabled.")
    
    def on_toggle_inspection(self):
        """Controls the defect detection logic and automated movement."""
        if self.btn_inspect.isChecked():
            self.btn_inspect.setText("STOP INSPECTION")
            self.btn_inspect.setStyleSheet("background-color: #c0392b; font-weight: bold;")
            
            self.btn_scan.setEnabled(False)
            self.btn_scan.setStyleSheet("background-color: #555; color: #888; font-weight: bold;")
            
            self.thread.start_inspection_sequence()
        else:
            self.btn_inspect.setText("Start Inspection")
            self.btn_inspect.setStyleSheet("background-color: #d35400; font-weight: bold;")
            
            self.btn_scan.setEnabled(True)
            self.btn_scan.setStyleSheet("background-color: #27ae60; font-weight: bold;")
            
            self.thread.stop_inspection_sequence()
            self.actuator.stop_actuator(2)
            self.actuator.stop_actuator(1)
    
    def on_toggle_reasoner(self):
        """Manual trigger for the LLM to analyze current defect data."""
        if self.btn_reasoner.isChecked():
            self.btn_reasoner.setText("Analyzing...")
            self.btn_reasoner.setStyleSheet("background-color: #ff9900; color: black;")
            self.btn_reasoner.setEnabled(False) 

            # Pull current state from the thread
            inspection_data = self.thread.get_defect_statistics()

            # Logging raw JSON for debugging the prompt input
            self.update_reasoning_log(json.dumps(inspection_data, indent=2))
            self.update_reasoning_log(f"<b>DATA SENT:</b> Found {inspection_data['total_defects_found']} defects...")

            self.reasoner_thread.set_data(inspection_data)
            self.reasoner_thread.start()
        else:
            self.btn_reasoner.setText("Start Reasoner")
            self.btn_reasoner.setStyleSheet("background-color: #444; color: white;")
    
    def handle_reasoner_output(self, data_dict):
        """Callback when LLM finishes. Resumes hardware if it was waiting on the LLM."""
        llmMessage = f"\n<b>LLM OUTPUT:</b>\n{json.dumps(data_dict, indent=2)}\n"
        
        self.update_reasoning_log(llmMessage)
        self.update_reasoning_log("-" * 30)

        # If inspection was paused for a 'thinking' step, resume now
        if self.thread.scan_state == self.thread.WAIT_FOR_LLM:
            self.thread.resume_inspection()
        
        self.btn_reasoner.setText("Start Reasoner")
        self.btn_reasoner.setEnabled(True)

    # --- UI LOGIC / FILTER SYNCING ---
    def on_min_dist_change(self, val):
        self.spin_min.setValue(val / 10.0)
        self.update_software_filters()
    def on_min_spin_change(self, val):
        self.slider_min.setValue(int(val * 10))
        self.update_software_filters()
    def on_max_dist_change(self, val):
        self.spin_max.setValue(val / 10.0)
        self.update_software_filters()
    def on_max_spin_change(self, val):
        self.slider_max.setValue(int(val * 10))
        self.update_software_filters()

    def update_software_filters(self):
        """Push new clipping planes to the RealSense thread."""
        min_d = self.spin_min.value()
        max_d = self.spin_max.value()
        if min_d < max_d:
            self.thread.update_filters(min_d, max_d)

    def on_disp_shift_change(self, val):
        self.spin_disp.setValue(val)
        self.update_hardware_params()
    def on_disp_spin_change(self, val):
        self.slider_disp.setValue(int(val))
        self.update_hardware_params()
    def on_units_change(self, val):
        self.spin_units.setValue(val)
        self.update_hardware_params()
    def on_units_spin_change(self, val):
        self.slider_units.setValue(int(val))
        self.update_hardware_params()

    def update_hardware_params(self):
        """Update internal RealSense ASIC parameters (Disparity, Units)."""
        disp = self.spin_disp.value()
        units = self.spin_units.value()
        self.thread.update_advanced_params(disp, units)

    def on_complete_tuning(self):
        """Locks the sliders and unlocks the operation buttons (Scan/Inspect)."""
        if hasattr(self, 'tile_5_stack'):
            self.tile_5_stack.setCurrentIndex(1)
        
        # Disable tuning widgets to prevent accidental changes during scan
        self.slider_min.setEnabled(False); self.spin_min.setEnabled(False)
        self.slider_max.setEnabled(False); self.spin_max.setEnabled(False)
        self.slider_disp.setEnabled(False); self.spin_disp.setEnabled(False)
        self.slider_units.setEnabled(False); self.spin_units.setEnabled(False)
        
        self.btn_complete.setEnabled(False)
        self.btn_complete.setText("Tuning Locked")
        self.btn_complete.setStyleSheet("background-color: #444; color: #888; font-weight: bold;")

        self.btn_scan.setEnabled(True)
        self.btn_scan.setStyleSheet("background-color: #27ae60; font-weight: bold;")
        self.update_log("SYSTEM: Tuning Complete. 3D Scanning Enabled.")

        self.btn_reasoner.setEnabled(True)
        self.btn_reasoner.setStyleSheet("background-color: #9b59b6; font-weight: bold;")
    
    def update_log(self, message: str):
        """Standard green terminal log in Tile 2."""
        self.log_console.append(message)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # --- IMAGE RENDERING SLOTS ---
    @pyqtSlot(QImage)
    def update_tile_3_rgb(self, qt_image):
        scaled = QPixmap.fromImage(qt_image).scaled(
            self.tile_3_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.tile_3_label.setPixmap(scaled)

    @pyqtSlot(QImage)
    def update_tile_4_defect(self, qt_image):
        scaled = QPixmap.fromImage(qt_image).scaled(
            self.tile_4_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.tile_4_label.setPixmap(scaled)

    @pyqtSlot(QImage)
    def update_tile_5_depth(self, qt_image):
        if hasattr(self, 'lbl_depth_feed'):
             scaled = QPixmap.fromImage(qt_image).scaled(
                self.lbl_depth_feed.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
             self.lbl_depth_feed.setPixmap(scaled)
    
    @pyqtSlot(QImage)
    def update_tile_5_pcd(self, q_img):
        """Switches stack to PCD view and displays the 3D render."""
        if self.tile_5_stack.currentIndex() != 1:
            self.tile_5_stack.setCurrentIndex(1)
        self.lbl_pcd_feed.setPixmap(QPixmap.fromImage(q_img))
    
    @pyqtSlot(str)
    def update_reasoning_log(self, text):
        """Blue-themed log for LLM text."""
        self.reasoning_log.append(text)
        sb = self.reasoning_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(dict)
    def update_legend(self, class_names):
        """Generates dynamic labels based on what YOLO/Classification model detects."""
        css_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF"]
        
        # Clear existing layout
        while self.tile_3_legend_layout.count():
            child = self.tile_3_legend_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        header = QLabel("Components:")
        header.setStyleSheet("font-weight: bold; text-decoration: underline; margin-top: 5px;")
        self.tile_3_legend_layout.addWidget(header)

        for cls_id in sorted(class_names.keys()):
            name = class_names[cls_id]
            color_hex = css_colors[cls_id % len(css_colors)]
            row_widget = QWidget(); row_layout = QHBoxLayout(); row_layout.setContentsMargins(0,2,0,2)
            color_box = QLabel(); color_box.setFixedSize(12, 12); color_box.setStyleSheet(f"background-color: {color_hex}; border: 1px solid white;")
            lbl_name = QLabel(f"{name}"); lbl_name.setStyleSheet("font-size: 10pt;")
            row_layout.addWidget(color_box); row_layout.addWidget(lbl_name); row_layout.addStretch()
            row_widget.setLayout(row_layout); self.tile_3_legend_layout.addWidget(row_widget)
        
        self.tile_3_legend_layout.addSpacing(15)
        mat_header = QLabel("Material Analysis:")
        mat_header.setStyleSheet("font-weight: bold; text-decoration: underline; margin-top: 5px;")
        self.tile_3_legend_layout.addWidget(mat_header)
        
        self.lbl_plastic_stat = QLabel("Plastic: 0.0%")
        self.lbl_copper_stat = QLabel("Copper: 0.0%")
        self.lbl_plastic_stat.setStyleSheet("color: #ccc;")
        self.lbl_copper_stat.setStyleSheet("color: #ffa500;")
        self.tile_3_legend_layout.addWidget(self.lbl_plastic_stat)
        self.tile_3_legend_layout.addWidget(self.lbl_copper_stat)
        self.tile_3_legend_layout.addStretch()

    @pyqtSlot(dict)
    def update_material_stats(self, stats):
        if self.lbl_plastic_stat:
            self.lbl_plastic_stat.setText(f"Plastic: {stats['plastic_pct']:.1f}%")
            self.lbl_copper_stat.setText(f"Copper: {stats['copper_pct']:.1f}%")
    
    @pyqtSlot(dict)
    def run_reasoner_auto(self, defect_data):
        """Triggered by worker thread when a full sweep is done and metrics are ready."""
        self.update_log(f"SYSTEM: Auto-analyzing defect via LLM...")
        self.tile_6.findChild(QTextEdit).append(f"<b>INPUT:</b> {json.dumps(defect_data)}")
        self.reasoner_thread.set_data(defect_data)
        self.reasoner_thread.start()

    # --- TILE CONFIGURATION (GUI CONSTRUCTION) ---
    def setup_tile_1(self):
        group = QGroupBox("1. Depth Tuning Controls")
        main_layout = QVBoxLayout()

        lbl_soft = QLabel("--- Software Clipping (Meters) ---")
        lbl_soft.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_soft.setStyleSheet("color: #aaa; font-weight: bold;")
        main_layout.addWidget(lbl_soft)

        row_min = QHBoxLayout(); row_min.addWidget(QLabel("Min:"))
        self.slider_min = QSlider(Qt.Orientation.Horizontal); self.slider_min.setRange(0, 100)
        self.spin_min = QDoubleSpinBox(); self.spin_min.setRange(0.0, 10.0); self.spin_min.setValue(0.1); self.spin_min.setSingleStep(0.05)
        row_min.addWidget(self.slider_min); row_min.addWidget(self.spin_min)
        
        row_max = QHBoxLayout(); row_max.addWidget(QLabel("Max:"))
        self.slider_max = QSlider(Qt.Orientation.Horizontal); self.slider_max.setRange(0, 100)
        self.spin_max = QDoubleSpinBox(); self.spin_max.setRange(0.0, 10.0); self.spin_max.setValue(0.4); self.spin_max.setSingleStep(0.05)
        row_max.addWidget(self.slider_max); row_max.addWidget(self.spin_max)

        main_layout.addLayout(row_min); main_layout.addLayout(row_max)
        self.slider_min.valueChanged.connect(self.on_min_dist_change)
        self.spin_min.valueChanged.connect(self.on_min_spin_change)
        self.slider_max.valueChanged.connect(self.on_max_dist_change)
        self.spin_max.valueChanged.connect(self.on_max_spin_change)

        main_layout.addSpacing(10)
        lbl_hard = QLabel("--- Hardware Advanced Params ---")
        lbl_hard.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_hard.setStyleSheet("color: #aaa; font-weight: bold;")
        main_layout.addWidget(lbl_hard)

        row_disp = QHBoxLayout(); row_disp.addWidget(QLabel("Disp Shift:"))
        self.slider_disp = QSlider(Qt.Orientation.Horizontal); self.slider_disp.setRange(0, 512)
        self.spin_disp = QSpinBox(); self.spin_disp.setRange(0, 512); self.spin_disp.setValue(100)
        row_disp.addWidget(self.slider_disp); row_disp.addWidget(self.spin_disp)

        row_units = QHBoxLayout(); row_units.addWidget(QLabel("Depth Units:"))
        self.slider_units = QSlider(Qt.Orientation.Horizontal); self.slider_units.setRange(100, 5000)
        self.spin_units = QSpinBox(); self.spin_units.setRange(100, 5000); self.spin_units.setValue(100)
        row_units.addWidget(self.slider_units); row_units.addWidget(self.spin_units)

        main_layout.addLayout(row_disp); main_layout.addLayout(row_units)
        self.slider_disp.valueChanged.connect(self.on_disp_shift_change)
        self.spin_disp.valueChanged.connect(self.on_disp_spin_change)
        self.slider_units.valueChanged.connect(self.on_units_change)
        self.spin_units.valueChanged.connect(self.on_units_spin_change)

        main_layout.addStretch()
        
        self.btn_complete = QPushButton("Complete Depth Tuning")
        self.btn_complete.setMinimumHeight(35)
        self.btn_complete.setStyleSheet("background-color: #2a82da; font-weight: bold;")
        self.btn_complete.clicked.connect(self.on_complete_tuning)
        main_layout.addWidget(self.btn_complete)

        main_layout.addSpacing(5)
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("color: #555;")
        main_layout.addWidget(line)

        self.btn_scan = QPushButton("Start 3D Scan")
        self.btn_scan.setCheckable(True)
        self.btn_scan.setMinimumHeight(40) 
        self.btn_scan.setEnabled(False)
        self.btn_scan.setStyleSheet("background-color: #555; color: #888; font-weight: bold;") 
        self.btn_scan.clicked.connect(self.on_toggle_scan)
        main_layout.addWidget(self.btn_scan)

        main_layout.addSpacing(5) 
        
        self.btn_inspect = QPushButton("Start Inspection")
        self.btn_inspect.setCheckable(True)
        self.btn_inspect.setMinimumHeight(40) 
        self.btn_inspect.setEnabled(False) 
        self.btn_inspect.setStyleSheet("background-color: #555; color: #888; font-weight: bold;")
        self.btn_inspect.clicked.connect(self.on_toggle_inspection)
        main_layout.addWidget(self.btn_inspect)
        
        main_layout.addSpacing(5)

        self.btn_reasoner = QPushButton("Start Reasoner")
        self.btn_reasoner.setCheckable(True)
        self.btn_reasoner.setMinimumHeight(40)
        self.btn_reasoner.setEnabled(False) 
        self.btn_reasoner.setStyleSheet("background-color: #555; color: #888; font-weight: bold;")
        self.btn_reasoner.clicked.connect(self.on_toggle_reasoner)
        main_layout.addWidget(self.btn_reasoner)

        group.setLayout(main_layout)
        return group

    def setup_tile_2(self):
        group = QGroupBox("2. System Logs")
        layout = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #111; color: #00FF00; font-family: Consolas, monospace;")
        layout.addWidget(self.log_console)
        group.setLayout(layout)
        return group

    def setup_tile_3(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        frame.setStyleSheet("background-color: #222; color: #fff;")
        h_layout = QHBoxLayout()
        video_label = QLabel("Initializing RGB + YOLO...")
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_label.setScaledContents(False)
        
        legend_frame = QFrame()
        legend_frame.setFixedWidth(160)
        legend_frame.setStyleSheet("background-color: #333; border-left: 1px solid #555;")
        legend_layout = QVBoxLayout()
        legend_label = QLabel("Waiting for Model...")
        legend_layout.addWidget(legend_label)
        legend_frame.setLayout(legend_layout)
        
        h_layout.addWidget(video_label); h_layout.addWidget(legend_frame); h_layout.setStretch(0, 1) 
        frame.setLayout(h_layout)
        return frame, video_label, legend_layout

    def setup_tile_4(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        frame.setStyleSheet("background-color: #222; color: #fff;")
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        label = QLabel("DEFECT DETECTION FEED")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(label)
        frame.setLayout(layout)
        return frame, label

    def setup_tile_5(self):
        stack = QStackedWidget()
        
        # Index 0: Raw Depth
        self.depth_view_container = QFrame()
        self.depth_view_container.setStyleSheet("background-color: #222; color: #fff;")
        layout_depth = QVBoxLayout()
        self.lbl_depth_feed = QLabel("Depth Feed")
        self.lbl_depth_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_depth.addWidget(self.lbl_depth_feed)
        self.depth_view_container.setLayout(layout_depth)
        
        # Index 1: Merged Point Cloud
        self.pcd_container = QFrame()
        self.pcd_container.setStyleSheet("background-color: #000; border: 1px solid #444;")
        layout_3d = QVBoxLayout()
        self.lbl_pcd_feed = QLabel("Waiting for 3D Scan...")
        self.lbl_pcd_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_pcd_feed.setScaledContents(True) 
        layout_3d.addWidget(QLabel("MERGED POINT CLOUD"))
        layout_3d.addWidget(self.lbl_pcd_feed)
        self.pcd_container.setLayout(layout_3d)
        
        stack.addWidget(self.depth_view_container)
        stack.addWidget(self.pcd_container)
        
        self.tile_5_stack = stack 
        return stack

    def setup_tile_6(self):
        group = QGroupBox("6. Reasoner Output")
        layout = QVBoxLayout()
        self.reasoning_log = QTextEdit()
        self.reasoning_log.setStyleSheet("background-color: #0e0e0e; color: #00e5ff; font-family: Consolas;")
        self.reasoning_log.setReadOnly(True)
        layout.addWidget(self.reasoning_log)
        group.setLayout(layout)
        return group