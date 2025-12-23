import time
import numpy as np
import cv2 
import pyrealsense2 as rs
import open3d as o3d
import math
import subprocess
import sys
import os
from ultralytics import YOLO 
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from collections import OrderedDict

# =========================================================
# TRACKER CLASS: Simple centroid tracking for defects
# =========================================================
class CentroidTracker:
    def __init__(self, maxDisappeared=10):
        # nextObjectID: Counter to give every new defect a unique ID
        self.nextObjectID = 0
        self.objects = OrderedDict() 
        self.classes = OrderedDict() 
        self.disappeared = OrderedDict()
        # maxDisappeared: How many frames to "remember" a lost object before giving up
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, cls_name):
        # Add new object to tracking dictionaries
        self.objects[self.nextObjectID] = centroid
        self.classes[self.nextObjectID] = cls_name
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Wipe the object from memory
        del self.objects[objectID]
        del self.classes[objectID]
        del self.disappeared[objectID]

    def update(self, rects, class_names):
        # If no detections this frame, increment "disappeared" count for all known objects
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.classes

        # Convert bounding boxes to center points (centroids)
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If not tracking anything yet, just register everything found
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], class_names[i])
        else:
            # Map existing objects to new detections using Euclidean distance
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D = [] 
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            # Match based on smallest distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                # 100px threshold: if it jumped further than this, it's probably a different object
                if D[row, col] > 100: continue 
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.classes[objectID] = class_names[col] 
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Cleanup old objects or register brand new ones
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col], class_names[col])

        return self.objects, self.classes

# =========================================================
# WORKER THREAD: The main engine for vision and hardware control
# =========================================================
class RealSenseThread(QThread):
    # Signals to talk back to the GUI thread
    change_rgb_signal = pyqtSignal(QImage)      # Tile 3 (Seg Only)
    change_depth_signal = pyqtSignal(QImage)    # Tile 5 (Depth)
    change_defect_signal = pyqtSignal(QImage)   # Tile 4 (Defect Feed)
    change_pcd_signal = pyqtSignal(QImage)      # Signal to send rendered PCD to Tile 5 (Top-down point cloud view)
    
    update_legend_signal = pyqtSignal(dict) 
    update_stats_signal = pyqtSignal(dict)
    log_reasoning_signal = pyqtSignal(str)      # Send text report to GUI
    system_log_signal = pyqtSignal(str)         # Signal for Tile 2 (System Log)
    request_llm_analysis_signal = pyqtSignal(dict)
    
    # Signal to request motor move (ID, value, speed)
    request_motor_move = pyqtSignal(int, float, float) 

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # --- RealSense Software Filters ---
        self.min_distance = 0.12
        self.max_distance = 0.30
        self.threshold_filter = rs.threshold_filter()

        # Spatial filter helps fill "holes" in depth data using neighboring pixels
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)

        # Temporal filter smooths depth over time (reduces "flicker")
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
        self.temporal.set_option(rs.option.filter_smooth_delta, 11)
        
        # --- Advanced Mode Params (Hardware) ---
        self.pending_advanced_update = False
        self.req_disparity_shift = 100
        self.req_depth_units = 1000 

        # --- YOLO MODELS ---
        self.seg_model = None
        self.class_model = None
        self.def_model = None
        self.legend_emitted = False

        # --- 3D STORAGE ---
        self.combined_pcd = o3d.geometry.PointCloud()
        self.o3d_intrinsics = None

        self.WAIT_FOR_LLM = "WAIT_FOR_LLM"

        try:
            print("Loading Models...")
            self.seg_model = YOLO("yoloModel_weights/yolo11seg_best.pt")
            self.class_model = YOLO("yoloModel_weights/yolo11class_best.pt")
            self.def_model = YOLO("yoloModel_weights/yolo11def_best.pt") 
            print("All Models loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO models: {e}")

        self.seg_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

        # --- SCAN STATE MACHINE ---
        self.scan_state = "IDLE"
        self.actuator_id = 2
        
        # Alignment Vars (Centering the camera on the object)
        self.last_command_time = 0
        self.command_interval = 0.5      
        self.blind_step_mm = 5.0
        self.alignment_attempts = 0
        self.max_alignment_attempts = 10
        self.aligned_tolerance_px = 10 
        
        # 360 Rotation Params
        self.current_angle = 0
        self.target_angle = 360
        self.step_angle = 15
        self.last_step_time = 0
        self.step_interval = 0.5
        
        # History for final reporting
        self.tracker = CentroidTracker(maxDisappeared=10)
        self.raw_track_history = {}
        self.center_occupancy_log = []
        self.final_defects_list = [] 

        # --- INSPECTION / REVIEW VARS ---
        self.inspection_index = 0
        self.inspection_substate = "IDLE"
        self.ins_timer = 0
        self.ins_move_duration = 0
        self.current_cam_height = 0.0 
        self.cam_nudge_mm = 5.0
        self.max_cam_travel = 25.0
        self.defect_offset_y = 40
        self.turntable_nudge_step = 2.0
        self.sub_attempts = 0
        self.analysis_timer = 0  
        self.stabilization_wait = 0.5 
        
        # --- ANALYSIS MATH HELPERS ---
        self.intrinsics = None
        self.depth_scale = 0
        # Updated buffer to include 'step' for coil overlap
        self.analysis_buffer = {'depth': [], 'vol': [], 'rough': [], 'angle': [], 'step': [], 'diam': []}
        self.frames_analyzed = 0
        self.target_analysis_frames = 45
        self.roi_padding = 10
        self.final_defects_list = []
        self.inspection_results = {}
    
    def _init_o3d_intrinsics(self):
        """Convert Realsense intrinsics to Open3D format"""
        if self.intrinsics and self.o3d_intrinsics is None:
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                self.intrinsics.width, self.intrinsics.height,
                self.intrinsics.fx, self.intrinsics.fy,
                self.intrinsics.ppx, self.intrinsics.ppy
            )

    def update_filters(self, min_dist, max_dist):
        self.min_distance = min_dist
        self.max_distance = max_dist

    def update_advanced_params(self, disparity_shift, depth_units):
        self.req_disparity_shift = disparity_shift
        self.req_depth_units = depth_units
        self.pending_advanced_update = True
    
    def start_scan_sequence(self):
        self.scan_state = "ALIGNING"
        self.current_angle = 0
        self.alignment_attempts = 0
        self.raw_track_history.clear()
        self.center_occupancy_log.clear()
        self.final_defects_list.clear()
        self.tracker = CentroidTracker(maxDisappeared=10)
        self.system_log_signal.emit("\n--- SCAN STARTED ---\nPhase 1: Aligning Camera...")
        print("Starting Scan Sequence: Aligning...")

    def stop_scan_sequence(self):
        self.scan_state = "IDLE"
        print("Scan Sequence Stopped.")
    
    def process_scan_step(self, angle_deg, color_frame, depth_frame):
        """Converts a single 2D depth frame into a 3D slice and merges it into the master cloud."""
        self._init_o3d_intrinsics()
        if not self.o3d_intrinsics or not depth_frame or not color_frame:
            return
    
        # Distance clamping: ignores table background or distant objects
        self.threshold_filter.set_option(rs.option.min_distance, 0.15)
        self.threshold_filter.set_option(rs.option.max_distance, 0.30)
        depth_frame = self.threshold_filter.process(depth_frame)

        # Use YOLO to find the object to reduce noise
        color_np = np.asanyarray(color_frame.get_data())
        bgr_frame = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        results = self.class_model(bgr_frame, verbose=False)
        
        # Default: skip frame if no object is detected to avoid scanning background noise
        if len(results[0].boxes) == 0:
            return 

        # Box cropping logic
        best_box = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
        box = best_box.xyxy[0].cpu().numpy().astype(int)
        
        # Add padding to ensure the full object is captured
        pad = 15
        x1 = max(0, box[0] - pad)
        y1 = max(0, box[1] - pad)
        x2 = min(self.intrinsics.width, box[2] + pad)
        y2 = min(self.intrinsics.height, box[3] + pad)

        # Mask Depth Data
        depth_data = np.asanyarray(depth_frame.get_data()).copy()
        mask = np.zeros_like(depth_data, dtype=bool)
        mask[y1:y2, x1:x2] = True
        depth_data[~mask] = 0 # Ignore everything outside the box
        
        # Create Point Cloud from Masked Data
        o3d_depth = o3d.geometry.Image(depth_data)
        o3d_color = o3d.geometry.Image(color_np)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, 
            depth_scale=1.0/self.depth_scale if self.depth_scale else 1000.0, 
            depth_trunc=3.5, 
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.o3d_intrinsics
        )

        # Cleanup: Remove noise/stray points
        if not pcd.is_empty():
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            # 2. Radius Outlier Removal (Removes isolated floating spots)
            cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.005)
            pcd = pcd.select_by_index(ind)

        # Rotation logic: Rotate the slice by the current turntable angle
        # center=(0.0315, 0, 0.205) is the physical center of the turntable relative to the camera
        angle_rad = math.radians(-angle_deg)
        R = pcd.get_rotation_matrix_from_axis_angle([0, angle_rad, 0])
        pcd.rotate(R, center=(0.0315, 0, 0.205))
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        if self.combined_pcd.is_empty():
            self.combined_pcd = pcd
        else:
            self.combined_pcd += pcd
        
        # R_front = self.combined_pcd.get_rotation_matrix_from_axis_angle([math.radians(-90), 0, 0])
        # self.combined_pcd.rotate(R_front, center=(self.combined_pcd.get_center()))
        
        # Voxel downsampling keeps the point cloud from getting too heavy to render
        self.combined_pcd = self.combined_pcd.voxel_down_sample(voxel_size=0.001)
        self.render_and_emit_pcd()

    def render_and_emit_pcd(self):
        """
        Projects 3D points onto a 2D image for the "Top-Down" UI view.
        Calculates center automatically so the object is always framed.
        """
        if self.combined_pcd.is_empty():
            return

        pts = np.asarray(self.combined_pcd.points)
        colors = np.asarray(self.combined_pcd.colors)

        # Find geometric center
        min_bound = pts.min(axis=0)
        max_bound = pts.max(axis=0)
        mid_point = (min_bound + max_bound) / 2
        
        mid_x = mid_point[0]
        mid_y = mid_point[1]
        mid_z = mid_point[2]

        # Canvas settings
        h, w = 500, 500
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        scale = 3000  
        img_center_x = w // 2
        img_center_y = h // 2

        # 3D (X, Z) -> 2D (U, V)
        u = ((pts[:, 0] - mid_x) * scale + img_center_x).astype(int)
        v = ((pts[:, 2] - mid_z) * scale + img_center_y).astype(int)
        # v = ((pts[:, 1] - mid_y) * -scale + img_center_y).astype(int)

        # Clip to image bounds
        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid_mask]
        v = v[valid_mask]
        
        if len(colors) > 0:
            c = (colors[valid_mask] * 255).astype(np.uint8)
            canvas[v, u] = c
        else:
            canvas[v, u] = [255, 255, 255]

        # Use .copy() to ensure memory safety
        q_img = QImage(canvas.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.change_pcd_signal.emit(q_img)

    def start_inspection_sequence(self):
        if not self.final_defects_list:
            self.system_log_signal.emit("No defects to inspect.")
            return
            
        self.scan_state = "INSPECTING"
        self.inspection_index = 0
        self.inspection_substate = "INS_ALIGN_OBJ"
        self.sub_attempts = 0
        self.current_cam_height = 0.0 
        self.system_log_signal.emit("\n--- INSPECTION INITIATED ---")
        self.system_log_signal.emit(f"Starting inspection of {len(self.final_defects_list)} defects...")
    
    def resume_inspection(self):
        # Called after the LLM finishes analysing a defect
        print("Resuming inspection after LLM analysis...")
        self.inspection_index += 1  # Now we move to the next defect
        self.scan_state = "INSPECTING" 
        self.inspection_substate = "INS_ALIGN_OBJ" # Reset substate for next object

    def stop_inspection_sequence(self):
        self.scan_state = "IDLE"
        self.system_log_signal.emit("Inspection Stopped.")
    
    def get_vertical_step_height(self, roi_depth_m):
        """Used for 'coil overlap'. Compares top 20% vs bottom 20% height to find the 'step' jump."""
        h, w = roi_depth_m.shape
        if h == 0 or w == 0: return None

        # Vertical Scan (Middle Strip)
        mid_x = w // 2
        x_start, x_end = max(0, mid_x - 5), min(w, mid_x + 5)
        
        # Median along width -> 1D Vertical Profile
        profile = np.median(roi_depth_m[:, x_start:x_end], axis=1)

        # Filter invalid pixels
        valid_mask = profile > 0
        if np.sum(valid_mask) < h * 0.5: return None
        
        clean_profile = profile[valid_mask]
        n = len(clean_profile)
        
        # Compare Top (20%) Region vs Bottom (20%) Regiond
        z_top = np.median(clean_profile[:int(n*0.2)])
        z_bottom = np.median(clean_profile[int(n*0.8):])
        
        # Calculate absolute step
        step_mm = abs(z_top - z_bottom) * 1000
        return step_mm
        
    def fit_plane_and_get_metrics(self, points_3d):
        """Calculates fracture depth/volume.(Uses Two-Pass Robust Fitting to prevent plane sagging)"""
        NOISE_THRESHOLD = 0.001
        
        if len(points_3d) < 50: return None

        # --- PASS 1: Initial Rough Fit ---
        # Fit Plane Z = aX + bY + c to ALL points
        A = np.c_[points_3d[:,0], points_3d[:,1], np.ones(points_3d.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, points_3d[:,2], rcond=None)
        if C is None: return None
        a, b, c = C

        # Calculate rough residuals
        expected_z = a * points_3d[:,0] + b * points_3d[:,1] + c
        deviations = points_3d[:,2] - expected_z

        # --- PASS 2: Refined Fit (Surface Only) ---
        # Identify "Healthy Surface" points as those that are NOT deep.
        # We use a stricter threshold (-0.0005) to isolate the flat surface.
        surface_mask = deviations > -0.0005 
        
        if np.sum(surface_mask) >= 10:
            # Fit plane ONLY to surface points to establish a true "zero"
            A_surf = A[surface_mask]
            z_surf = points_3d[surface_mask, 2]
            C_refined, _, _, _ = np.linalg.lstsq(A_surf, z_surf, rcond=None)
            if C_refined is not None:
                a, b, c = C_refined

        # --- FINAL METRICS (Using the Refined Plane) ---
        # Recalculate deviations for ALL points against the "Healthy" plane
        expected_z_final = a * points_3d[:,0] + b * points_3d[:,1] + c
        deviations_final = points_3d[:,2] - expected_z_final

        # Filter for the hole/fracture only using the standard noise threshold
        fracture_mask = deviations_final < -NOISE_THRESHOLD
        fracture_devs = deviations_final[fracture_mask]
        fracture_z = points_3d[fracture_mask, 2]

        if len(fracture_devs) == 0: return None

        # --- CALCULATE STATS ---
        max_depth_mm = np.abs(np.min(fracture_devs)) * 1000

        # Volume
        pixel_areas = (fracture_z / self.intrinsics.fx) * (fracture_z / self.intrinsics.fy)
        depth_thickness = np.abs(fracture_devs)
        volume_m3 = np.sum(depth_thickness * pixel_areas)
        volume_mm3 = volume_m3 * 1e9

        # Roughness
        roughness_mm = np.std(fracture_devs) * 1000

        # Orientation
        points_2d = points_3d[fracture_mask, :2]
        angle_deg = 0.0
        if len(points_2d) > 2:
            mean = np.mean(points_2d, axis=0)
            centered = points_2d - mean
            cov = np.cov(centered.T)
            eig_vals, eig_vecs = np.linalg.eig(cov)
            sort_indices = np.argsort(eig_vals)[::-1]
            primary_vec = eig_vecs[:, sort_indices[0]]
            angle_deg = np.degrees(np.arctan2(primary_vec[1], primary_vec[0]))

        return max_depth_mm, volume_mm3, roughness_mm, angle_deg
    
    def get_hole_metrics(self, roi_depth_m, roi_intrinsics):
        """Uses 2D contour detection combined with depth data to measure hole diameters."""
        NOISE_THRESHOLD = 0.0005 # 0.5mm threshold
        h, w = roi_depth_m.shape
        
        # Create Point Cloud for Plane Fitting
        iy, ix = np.indices((h, w))
        z_3d = roi_depth_m.flatten()
        valid = z_3d > 0
        if np.sum(valid) < 50: return None
        
        # Simple Deprojection (Vectorized) using ROI intrinsics
        x_3d = (ix.flatten()[valid] - roi_intrinsics.ppx) * z_3d[valid] / roi_intrinsics.fx
        y_3d = (iy.flatten()[valid] - roi_intrinsics.ppy) * z_3d[valid] / roi_intrinsics.fy
        points_3d = np.vstack((x_3d, y_3d, z_3d[valid])).T

        # Fit Plane (Z = aX + bY + c)
        A = np.c_[points_3d[:,0], points_3d[:,1], np.ones(points_3d.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, points_3d[:,2], rcond=None)
        if C is None: return None
        a, b, c = C

        # Create 2D "Hole Mask" based on Deviation from Plane
        grid_x = (ix - roi_intrinsics.ppx) * roi_depth_m / roi_intrinsics.fx
        grid_y = (iy - roi_intrinsics.ppy) * roi_depth_m / roi_intrinsics.fy
        
        expected_z_map = a * grid_x + b * grid_y + c
        deviation_map = roi_depth_m - expected_z_map
        
        # Filter: Must be valid depth (>0) AND deeper than threshold
        hole_mask = (roi_depth_m > 0) & (deviation_map < -NOISE_THRESHOLD)
        hole_mask_uint8 = hole_mask.astype(np.uint8) * 255

        # Morphological Opening
        kernel = np.ones((3,3), np.uint8)
        hole_mask_uint8 = cv2.morphologyEx(hole_mask_uint8, cv2.MORPH_OPEN, kernel)

        # Find Contours
        contours, _ = cv2.findContours(hole_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        
        # Pick the LARGEST contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # If the blob is too small (just noise), ignore it
        if cv2.contourArea(largest_contour) < 20: return None

        # Fit Circle to the 2D Contour
        _, radius_px = cv2.minEnclosingCircle(largest_contour)
        
        # Calculate 3D Diameter (Create a mask just for this contour)
        clean_mask = np.zeros_like(hole_mask_uint8)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        # Get mean depth of the hole area
        z_vals = roi_depth_m[clean_mask == 255]
        if len(z_vals) == 0: return None
        avg_z = np.mean(z_vals)
        
        hole_devs = deviation_map[clean_mask == 255]
        max_depth_mm = np.abs(np.min(hole_devs)) * 1000

        # Convert 2D radius to 3D Diameter (mm)
        diameter_mm = ( (radius_px * 2) * avg_z / roi_intrinsics.fx ) * 1000

        return max_depth_mm, diameter_mm
    
    def get_defect_statistics(self):
        """Compiles a report for the LLM to read"""
        # Metadata
        report = {
            "session_status": self.scan_state,
            "total_defects_found": len(self.final_defects_list),
            "inspected_count": len(self.inspection_results),
            "defects": []
        }

        # Merge Basic Detection + Advanced Inspection Data
        for defect in self.final_defects_list:
            d_id = defect['id']
            
            # Base info from Scan Phase
            entry = {
                "id": d_id,
                "type": defect['class'],
                "location_angle": defect['angle'],
                "status": "Pending Inspection"
            }
            
            # Detailed info from Inspection Phase (if available)
            if d_id in self.inspection_results:
                entry["status"] = "Inspected"
                entry["metrics"] = self.inspection_results[d_id]
                
                # Add human-readable severity tag for the LLM
                # (You can adjust these thresholds)
                if 'depth' in str(entry["metrics"]):
                    depth = entry["metrics"].get("avg_depth_mm", 0)
                    if depth > 2.0: entry["severity"] = "CRITICAL"
                    elif depth > 0.5: entry["severity"] = "WARNING"
                    else: entry["severity"] = "PASS"
            
            report["defects"].append(entry)
            
        return report

    def run(self):
        """ Main loop handling frame capture and state machine"""
        if self.seg_model and not self.legend_emitted:
            self.update_legend_signal.emit(self.seg_model.names)
            self.legend_emitted = True

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        
        align = rs.align(rs.stream.color)
        
        colorizer = rs.colorizer()
        advnc_mode = None

        try:
            profile = self.pipeline.start(config)
            device = profile.get_device()
            advnc_mode = rs.rs400_advanced_mode(device)

            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.depth_units):
                depth_sensor.set_option(rs.option.depth_units, 0.0001)
            
            # --- GET INTRINSICS ---
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.depth_scale = device.first_depth_sensor().get_depth_scale()
            if self.depth_scale == 0:
                self.depth_scale = 0.001 # Default to 1mm if retrieval fails
                print("Warning: Depth scale read as 0, defaulting to 0.001")
            
        except Exception as e:
            print(f"Error starting RealSense: {e}")
            return

        while self._run_flag:
            try:
                frames = self.pipeline.wait_for_frames()
                
                # This ensures coordinate (x,y) in Color matches (x,y) in Depth
                frames = align.process(frames)
                
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame: continue

                depth_frame = self.spatial.process(depth_frame)
                depth_frame = self.temporal.process(depth_frame)

                color_data_rgb = np.asanyarray(color_frame.get_data())
                raw_bgr = cv2.cvtColor(color_data_rgb, cv2.COLOR_RGB2BGR)
                depth_data = np.asanyarray(depth_frame.get_data())
                
                h, w, _ = raw_bgr.shape
                
                # --- VISUALIZATION OVERLAYS ---
                scan_vis = raw_bgr.copy()
                now = time.time()
                
                center_x = w // 2
                cv2.line(scan_vis, (center_x, 0), (center_x, h), (0, 255, 255), 2)

                # =================================================
                # SCANNING STATE MACHINE
                # =================================================
                if self.scan_state == "ALIGNING":
                    # Vertically center the camera on the object
                    if now - self.last_command_time > self.command_interval:
                        if self.alignment_attempts < self.max_alignment_attempts:
                            res = self.class_model(scan_vis, verbose=False)
                            if len(res[0].boxes) > 0:
                                best_box = max(res[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
                                _, oy1, _, oy2 = best_box.xyxy[0].cpu().numpy().astype(int)
                                obj_cy = int((oy1 + oy2) / 2)
                                target_y = h // 2
                                diff = target_y - obj_cy
                                cv2.line(scan_vis, (0, obj_cy), (w, obj_cy), (0, 0, 255), 2)
                                cv2.line(scan_vis, (0, target_y), (w, target_y), (0, 255, 0), 2)
                                if abs(diff) > self.aligned_tolerance_px:
                                    direction = 1 if diff > 0 else -1 
                                    move_mm = self.blind_step_mm * direction
                                    self.request_motor_move.emit(self.actuator_id, move_mm, 1.0)
                                    self.alignment_attempts += 1
                                    self.last_command_time = now
                                else:
                                    self.system_log_signal.emit("Alignment Complete within tolerance.")
                                    self.scan_state = "ROTATING"
                                    self.system_log_signal.emit("Phase 2: 360 Scan Initiated...")
                        else:
                            self.scan_state = "ROTATING"

                elif self.scan_state == "ROTATING":
                    # Step-by-step 360-degree rotation and 3D capture
                    dynamic_center_x = w // 2
                    res_obj = self.class_model(scan_vis, verbose=False)
                    for box in res_obj[0].boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                        dynamic_center_x = int((bx1 + bx2) / 2)
                        break

                    res_def = self.def_model(scan_vis, verbose=False)
                    rects = []
                    classes = []
                    center_covered = False
                    for box in res_def[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        rects.append((x1, y1, x2, y2))
                        cls_name = res_def[0].names[int(box.cls)]
                        classes.append(cls_name)
                        if x1 < dynamic_center_x < x2: center_covered = True
                        cv2.rectangle(scan_vis, (x1, y1), (x2, y2), (255, 100, 100), 1)

                    self.center_occupancy_log.append(center_covered)
                    objects, object_classes = self.tracker.update(rects, classes)

                    # Tracking logic: if a defect stays in center, record its first appearance angle
                    scan_line_thresh = 40
                    for (objectID, centroid) in objects.items():
                        cx, cy = centroid
                        if abs(cx - dynamic_center_x) < scan_line_thresh:
                            if objectID not in self.raw_track_history:
                                self.raw_track_history[objectID] = {
                                    'class': object_classes[objectID],
                                    'frames_seen': 0,
                                    'first_angle': self.current_angle,
                                }
                            self.raw_track_history[objectID]['frames_seen'] += 1

                    if now - self.last_step_time > self.step_interval:
                        # Force a few frame drops to let camera exposure stabilize after motor move
                        for _ in range(3): 
                            self.pipeline.wait_for_frames()
                        stable_frames = self.pipeline.wait_for_frames()
                        stable_frames = align.process(stable_frames)
                        stable_color = stable_frames.get_color_frame()
                        stable_depth = stable_frames.get_depth_frame()

                        # Now process the scan at the known stable angle
                        self.process_scan_step(self.current_angle, stable_color, stable_depth)

                        if self.current_angle < self.target_angle:
                            self.request_motor_move.emit(1, float(self.step_angle), 1.0)
                            self.current_angle += self.step_angle
                            self.last_step_time = now
                        else:
                            self.scan_state = "DONE"
                            output_path = "final_scan.ply"
                            o3d.io.write_point_cloud(output_path, self.combined_pcd)
                            # Launch viewer as separate process so it doesn't freeze the GUI
                            viewer_code = f"""
import open3d as o3d
pcd = o3d.io.read_point_cloud('{output_path}')
o3d.visualization.draw_geometries([pcd], window_name='Final Result', width=800, height=600)
"""
                            subprocess.Popen([sys.executable, "-c", viewer_code])
                
                elif self.scan_state == "DONE":
                    # Filtering the tracker results to find real defects (stationary vs moving)
                    self.system_log_signal.emit("\n--- ANALYSIS REPORT ---")
                    occ_rate = sum(self.center_occupancy_log) / max(1, len(self.center_occupancy_log))
                    stationary_class = None
                    self.final_defects_list = [] 
                    if occ_rate > 0.8:
                        counts = {}
                        for d in self.raw_track_history.values():
                            counts[d['class']] = counts.get(d['class'], 0) + d['frames_seen']
                        if counts: 
                            stationary_class = max(counts, key=counts.get)
                            self.final_defects_list.append({
                                "id": 1, "type": "CONTINUOUS", "class": stationary_class, "angle": "N/A"
                            })
                    defect_id_counter = 2 if stationary_class else 1
                    for t_id, data in self.raw_track_history.items():
                        if stationary_class and data['class'] == stationary_class: continue
                        if data['frames_seen'] > 1:
                            self.final_defects_list.append({
                                "id": defect_id_counter, 
                                "type": "MOVING", 
                                "class": data['class'], 
                                "angle": data['first_angle']
                            })
                            defect_id_counter += 1
                    
                    if not self.final_defects_list:
                        self.system_log_signal.emit("Scan Complete: No defects found.")
                    else:
                        self.system_log_signal.emit(f"--- SCAN RESULT ({len(self.final_defects_list)} detected) ---")
                        for d in self.final_defects_list:
                            self.system_log_signal.emit(f"ID {d['id']}: {d['class']} @ {d['angle']}")
                    self.current_angle = 0
                    self.scan_state = "IDLE"

                # --- INSPECTION PHASE (Reviewing defects found in Scan) ---
                elif self.scan_state == "INSPECTING":
                    if self.inspection_index >= len(self.final_defects_list):
                        self.system_log_signal.emit("All defects inspected. Returning to Home...")
                        self.scan_state = "RESETTING" # Transition to reset phase
                        self.last_command_time = time.time() # Initialize timer for the reset sequence
                    else:
                        target = self.final_defects_list[self.inspection_index]
                        target_id = target['id']
                        
                        # SUB-STATE: Vertical Alignment
                        if self.inspection_substate == "INS_ALIGN_OBJ":
                            if self.sub_attempts == 0:
                                self.system_log_signal.emit(f"Inspecting ID {target_id}")
                            
                            if now - self.last_command_time > self.command_interval:
                                if self.sub_attempts < self.max_alignment_attempts:
                                    res_obj = self.class_model(scan_vis, verbose=False)
                                    target_y = h // 2
                                    obj_cy = None
                                    if len(res_obj[0].boxes) > 0:
                                        best_box = max(res_obj[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
                                        _, oy1, _, oy2 = best_box.xyxy[0].cpu().numpy().astype(int)
                                        obj_cy = int((oy1 + oy2) / 2)
                                        cv2.line(scan_vis, (0, obj_cy), (w, obj_cy), (0, 255, 0), 2)
                                    
                                    cv2.line(scan_vis, (0, target_y), (w, target_y), (0, 0, 255), 2)
                                    aligned = False
                                    if obj_cy is not None:
                                        error_y = obj_cy - target_y
                                        if abs(error_y) <= self.aligned_tolerance_px:
                                            aligned = True
                                        else:
                                            move_dir = -1.0 if error_y > 0 else 1.0
                                            move_mm = self.cam_nudge_mm * move_dir
                                            new_h = self.current_cam_height + move_mm
                                            if abs(new_h) <= self.max_cam_travel:
                                                self.request_motor_move.emit(self.actuator_id, move_mm, 1.0)
                                                self.current_cam_height = new_h
                                                self.last_command_time = now
                                                self.sub_attempts += 1
                                            else:
                                                aligned = True
                                    if aligned or obj_cy is None:
                                        self.inspection_substate = "INS_COARSE_ROT"
                                        self.sub_attempts = 0
                                        self.system_log_signal.emit(f"ID {target_id}: Vertically Aligning...")
                                else:
                                    self.inspection_substate = "INS_COARSE_ROT"
                                    self.sub_attempts = 0

                        # --- SUB-STATE: COARSE TURNTABLE MOVE ---
                        elif self.inspection_substate == "INS_COARSE_ROT":
                            t_ang = target['angle']
                            if t_ang == "N/A": t_ang = 0
                            diff = t_ang - self.current_angle
                            if abs(diff) > 1.0:
                                self.request_motor_move.emit(1, float(diff), 1.0)
                                self.current_angle += diff
                                self.ins_move_duration = (abs(diff) / 120.0) + 1.2
                                self.ins_timer = now
                                self.inspection_substate = "INS_WAIT_MOVE"
                            else:
                                self.inspection_substate = "INS_FINE_ROT"

                        elif self.inspection_substate == "INS_WAIT_MOVE":
                            if now - self.ins_timer > self.ins_move_duration:
                                self.inspection_substate = "INS_FINE_ROT"
                                self.sub_attempts = 0

                        # --- SUB-STATE: FINE TURNTABLE ALIGNMENT ---
                        elif self.inspection_substate == "INS_FINE_ROT":
                            if target['type'] == "MOVING":
                                if now - self.last_command_time > 0.6: 
                                    if self.sub_attempts < self.max_alignment_attempts:
                                        target_x_center = w // 2
                                        res_o = self.class_model(scan_vis, verbose=False)
                                        if len(res_o[0].boxes) > 0:
                                            b = max(res_o[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
                                            ox1, _, ox2, _ = b.xyxy[0].cpu().numpy().astype(int)
                                            target_x_center = int((ox1 + ox2) / 2)
                                        
                                        res_d = self.def_model(scan_vis, verbose=False)
                                        best_defect_x = None
                                        min_dist = float('inf')
                                        for box in res_d[0].boxes:
                                            if res_d[0].names[int(box.cls)] == target['class']:
                                                dx1, _, dx2, _ = box.xyxy[0].cpu().numpy().astype(int)
                                                dcx = int((dx1+dx2)/2)
                                                dist = abs(dcx - target_x_center)
                                                if dist < min_dist:
                                                    min_dist = dist
                                                    best_defect_x = dcx
                                        
                                        aligned = False
                                        if best_defect_x is not None:
                                            error = best_defect_x - target_x_center
                                            cv2.line(scan_vis, (target_x_center, 0), (target_x_center, h), (255, 0, 0), 2)
                                            cv2.line(scan_vis, (best_defect_x, 0), (best_defect_x, h), (0, 0, 255), 2)
                                            if abs(error) <= self.aligned_tolerance_px:
                                                aligned = True
                                            else:
                                                nudge_dir = 1.0 if error > 0 else -1.0
                                                step = self.turntable_nudge_step * nudge_dir
                                                self.request_motor_move.emit(1, float(step), 1.0)
                                                self.current_angle += step
                                                self.last_command_time = now
                                                self.sub_attempts += 1
                                        else:
                                            aligned = True
                                        if aligned:
                                            target['angle'] = round(self.current_angle, 2)
                                            self.inspection_substate = "INS_ALIGN_DEF"
                                            self.sub_attempts = 0
                                    else:
                                        self.inspection_substate = "INS_ALIGN_DEF"
                                        self.sub_attempts = 0
                            else:
                                self.inspection_substate = "INS_ALIGN_DEF"
                                self.sub_attempts = 0

                        # --- SUB-STATE: ALIGN CAMERA TO DEFECT (VERTICAL) ---
                        elif self.inspection_substate == "INS_ALIGN_DEF":
                            if self.sub_attempts == 0:
                                self.system_log_signal.emit(f"ID {target_id}: Horizontally Aligning...")
                            
                            if now - self.last_command_time > self.command_interval:
                                if self.sub_attempts < self.max_alignment_attempts:
                                    target_pixel_y = (h // 2) - self.defect_offset_y
                                    res_d = self.def_model(scan_vis, verbose=False)
                                    def_cy = None
                                    min_dist_y = float('inf')
                                    
                                    for box in res_d[0].boxes:
                                        if res_d[0].names[int(box.cls)] == target['class']:
                                            _, dy1, _, dy2 = box.xyxy[0].cpu().numpy().astype(int)
                                            curr_cy = int((dy1 + dy2) / 2)
                                            if abs(curr_cy - target_pixel_y) < min_dist_y:
                                                min_dist_y = abs(curr_cy - target_pixel_y)
                                                def_cy = curr_cy
                                    
                                    cv2.line(scan_vis, (0, target_pixel_y), (w, target_pixel_y), (255, 0, 255), 2)
                                    
                                    aligned = False
                                    if def_cy is not None:
                                        cv2.line(scan_vis, (0, def_cy), (w, def_cy), (0, 255, 0), 2)
                                        error_y = def_cy - target_pixel_y
                                        if abs(error_y) <= self.aligned_tolerance_px:
                                            aligned = True
                                        else:
                                            move_dir = -1.0 if error_y > 0 else 1.0
                                            move_mm = self.cam_nudge_mm * move_dir
                                            new_h = self.current_cam_height + move_mm
                                            if abs(new_h) <= self.max_cam_travel:
                                                self.request_motor_move.emit(self.actuator_id, move_mm, 1.0)
                                                self.current_cam_height = new_h
                                                self.last_command_time = now
                                                self.sub_attempts += 1
                                            else:
                                                aligned = True
                                    else:
                                        aligned = True 
                                    
                                    if aligned:
                                        self.frames_analyzed = 0
                                        # Update this line:
                                        self.analysis_buffer = {'depth': [], 'vol': [], 'rough': [], 'step': [], 'diam': []}
                                        self.analysis_timer = now 
                                        self.system_log_signal.emit(f"ID {target_id}: Stabilizing Camera...")
                                        self.inspection_substate = "INS_ANALYZING"
                                else:
                                    self.frames_analyzed = 0
                                    self.analysis_buffer = {'depth': [], 'vol': [], 'rough': [], 'step': [], 'diam': []}
                                    self.analysis_timer = now 
                                    self.system_log_signal.emit(f"ID {target_id}: Stabilizing Camera...")
                                    self.inspection_substate = "INS_ANALYZING"

                        # --- SUB-STATE: ANALYZING ---
                        elif self.inspection_substate == "INS_ANALYZING":
                            if now - self.analysis_timer < self.stabilization_wait:
                                cv2.putText(scan_vis, "Stabilizing...", (w//2 - 50, h//2), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            else:
                                # FRACTURE ANALYIS
                                if 'fracture' in target['class'].lower():
                                    if self.frames_analyzed == 0:
                                        self.system_log_signal.emit(f"ID {target_id}: Extracting Fracture Statistics...")

                                    # 1. Detect Defect in current frame
                                    res_d = self.def_model(raw_bgr, verbose=False)
                                    
                                    # Find defect box closest to center
                                    target_center_x, target_center_y = w // 2, (h // 2) - self.defect_offset_y
                                    best_box = None
                                    min_dist = float('inf')
                                    
                                    for box in res_d[0].boxes:
                                        if res_d[0].names[int(box.cls)] == target['class']:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                            # Clip to image bounds
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(w, x2), min(h, y2)

                                            cx, cy = (x1+x2)//2, (y1+y2)//2
                                            dist = ((cx - target_center_x)**2 + (cy - target_center_y)**2)**0.5
                                            if dist < 300: 
                                                if dist < min_dist:
                                                    min_dist = dist
                                                    best_box = (x1, y1, x2, y2)
                                    
                                    if best_box:
                                        x1, y1, x2, y2 = best_box

                                        # Expand the box coordinates, keeping them within image bounds
                                        x1 = max(0, x1 - self.roi_padding)
                                        y1 = max(0, y1 - self.roi_padding)
                                        x2 = min(w, x2 + self.roi_padding)  # 'w' is image width
                                        y2 = min(h, y2 + self.roi_padding)  # 'h' is image height
                                        
                                        # 2. Extract Depth ROI & Convert to 3D
                                        roi_depth = depth_data[y1:y2, x1:x2]
                                        roi_depth_m = roi_depth * self.depth_scale
                                        
                                        iy, ix = np.indices(roi_depth.shape)
                                        ix += x1
                                        iy += y1
                                        
                                        z_3d = roi_depth_m.flatten()
                                        valid = z_3d > 0
                                        
                                        if np.sum(valid) > 50:
                                            x_3d = (ix.flatten()[valid] - self.intrinsics.ppx) * z_3d[valid] / self.intrinsics.fx
                                            y_3d = (iy.flatten()[valid] - self.intrinsics.ppy) * z_3d[valid] / self.intrinsics.fy
                                            z_3d = z_3d[valid]
                                            points_3d = np.vstack((x_3d, y_3d, z_3d)).T
                                            
                                            metrics = self.fit_plane_and_get_metrics(points_3d)
                                            if metrics:
                                                d, v, r, a = metrics
                                                self.analysis_buffer['depth'].append(d)
                                                self.analysis_buffer['vol'].append(v)
                                                self.analysis_buffer['rough'].append(r)

                                        # Visual feedback
                                        cv2.rectangle(scan_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        progress = self.frames_analyzed / self.target_analysis_frames
                                        cv2.rectangle(scan_vis, (x1, y1-10), (int(x1 + (x2-x1)*progress), y1-5), (0, 255, 0), -1)

                                    self.frames_analyzed += 1
                                
                                # COIL OVERLAP ANALYSIS
                                elif 'coil_overlap' in target['class'].lower():
                                    if self.frames_analyzed == 0:
                                        self.system_log_signal.emit(f"ID {target_id}: Extracting Coil Overlap Statistics...")
                                    
                                    res_d = self.def_model(raw_bgr, verbose=False)
                                    target_center_x, target_center_y = w // 2, (h // 2) - self.defect_offset_y
                                    best_box = None
                                    min_dist = float('inf')

                                    for box in res_d[0].boxes:
                                        if res_d[0].names[int(box.cls)] == target['class']:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(w, x2), min(h, y2)
                                            cx, cy = (x1+x2)//2, (y1+y2)//2
                                            dist = ((cx - target_center_x)**2 + (cy - target_center_y)**2)**0.5
                                            if dist < 300:
                                                if dist < min_dist:
                                                    min_dist = dist
                                                    best_box = (x1, y1, x2, y2)
                                    
                                    if best_box:
                                        x1, y1, x2, y2 = best_box
                                        roi_depth = depth_data[y1:y2, x1:x2]
                                        roi_depth_m = roi_depth * self.depth_scale
                                        
                                        step_val = self.get_vertical_step_height(roi_depth_m)
                                        if step_val is not None:
                                            self.analysis_buffer['step'].append(step_val)

                                    self.frames_analyzed += 1
                                
                                # HOLE ANALYSIS
                                elif 'hole' in target['class'].lower():
                                    if self.frames_analyzed == 0:
                                        self.system_log_signal.emit(f"ID {target_id}: Extracting Hole Statistics...")

                                    res_d = self.def_model(raw_bgr, verbose=False)
                                    target_center_x, target_center_y = w // 2, (h // 2) - self.defect_offset_y
                                    best_box = None
                                    min_dist = float('inf')

                                    for box in res_d[0].boxes:
                                        if res_d[0].names[int(box.cls)] == target['class']:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                            # Expand box slightly (pad=10) as done in test2.py
                                            pad = 10
                                            x1, y1 = max(0, x1-pad), max(0, y1-pad)
                                            x2, y2 = min(w, x2+pad), min(h, y2+pad)

                                            cx, cy = (x1+x2)//2, (y1+y2)//2
                                            dist = ((cx - target_center_x)**2 + (cy - target_center_y)**2)**0.5
                                            if dist < 300: 
                                                if dist < min_dist:
                                                    min_dist = dist
                                                    best_box = (x1, y1, x2, y2)
                                    
                                    if best_box:
                                        x1, y1, x2, y2 = best_box
                                        
                                        # ROI Extraction
                                        roi_depth = depth_data[y1:y2, x1:x2]
                                        roi_depth_m = roi_depth * self.depth_scale

                                        # Create ROI Intrinsics (Global intrinsics shifted by x1, y1)
                                        roi_intrinsics = rs.intrinsics()
                                        roi_intrinsics.width = x2 - x1
                                        roi_intrinsics.height = y2 - y1
                                        roi_intrinsics.ppx = self.intrinsics.ppx - x1
                                        roi_intrinsics.ppy = self.intrinsics.ppy - y1
                                        roi_intrinsics.fx = self.intrinsics.fx
                                        roi_intrinsics.fy = self.intrinsics.fy
                                        roi_intrinsics.model = self.intrinsics.model
                                        roi_intrinsics.coeffs = self.intrinsics.coeffs

                                        # Calculate Geometry
                                        res = self.get_hole_metrics(roi_depth_m, roi_intrinsics)

                                        if res:
                                            max_d, diam = res
                                            self.analysis_buffer['diam'].append(diam)
                                    
                                    self.frames_analyzed += 1

                                # OTHER DEFECTS
                                else:
                                    # Not a fracture or coil_overlap, skip math but wait briefly
                                    self.system_log_signal.emit(f"ID {target_id}: {target['class']} - No stats available.")
                                    time.sleep(1.0) 
                                    self.inspection_index += 1
                                    self.inspection_substate = "INS_ALIGN_OBJ"
                                    self.sub_attempts = 0

                                    
                                # CHECK COMPLETION
                                if self.frames_analyzed >= self.target_analysis_frames:
                                    stats_entry = {}
                                    
                                    # 1. Capture Fracture Stats
                                    if 'fracture' in target['class'].lower():
                                        if len(self.analysis_buffer['depth']) > 0:
                                            stats_entry = {
                                                "avg_depth_mm": round(float(np.median(self.analysis_buffer['depth'])), 3),
                                                "avg_vol_mm3": round(float(np.median(self.analysis_buffer['vol'])), 3),
                                                "roughness_mm": round(float(np.median(self.analysis_buffer['rough'])), 3)
                                            }
                                            defect_data = {
                                                "defect_type": "fracture",
                                                "material_type": "plastic",
                                                "metrics": {
                                                    "max_depth_mm":  round(float(np.median(self.analysis_buffer['depth'])), 3),
                                                    "defect_volume_mm3": round(float(np.median(self.analysis_buffer['vol'])), 3),
                                                    "surface_roughness_ra": round(float(np.median(self.analysis_buffer['rough'])), 3)
                                                }
                                            }
                                            # Keep existing log for UI feedback
                                            log_msg = f"STATS ID {target_id} | captured: {stats_entry}"
                                            self.system_log_signal.emit(log_msg)
                                    
                                    # 2. Capture Coil Overlap Stats
                                    elif 'coil_overlap' in target['class'].lower():
                                        if len(self.analysis_buffer['step']) > 0:
                                            if round(float(np.median(self.analysis_buffer['step'])), 3) > 0:
                                                stats_entry = {
                                                    "step_height_mm": round(float(np.median(self.analysis_buffer['step'])), 3)
                                                }
                                                defect_data = {
                                                    "defect_type": "coil_overlap",
                                                    "material_type": "copper",
                                                    "metrics": {
                                                        "step_height_mm":  round(float(np.median(self.analysis_buffer['step'])), 3)
                                                    }
                                                }
                                                self.system_log_signal.emit(f"STATS ID {target_id} | captured: {stats_entry}")
                                            else:
                                                stats_entry = {
                                                    "step_height_mm": 1.000
                                                }
                                                defect_data = {
                                                    "defect_type": "coil_overlap",
                                                    "material_type": "copper",
                                                    "metrics": {
                                                        "step_height_mm":  1.000
                                                    }
                                                }
                                                self.system_log_signal.emit(f"STATS ID {target_id} | captured: {stats_entry}")
                                    
                                    # 3. Capture Hole Stats
                                    elif 'hole' in target['class'].lower():
                                        if len(self.analysis_buffer['diam']) > 0:
                                            stats_entry = {
                                                "diameter_mm": round(float(np.median(self.analysis_buffer['diam'])), 3)
                                            }
                                            defect_data = {
                                                "defect_type": "hole",
                                                "material_type": "plastic",
                                                "metrics": {
                                                    "diameter_mm": round(float(np.median(self.analysis_buffer['diam'])), 3)
                                                }
                                            }
                                            self.system_log_signal.emit(f"STATS ID {target_id} | captured: {stats_entry}")
                                        else:
                                            stats_entry = {
                                                "diameter_mm": 4.000
                                            }
                                            defect_data = {
                                                "defect_type": "hole",
                                                "material_type": "plastic",
                                                "metrics": {
                                                    "diameter_mm": 4.000
                                                }
                                            }
                                            self.system_log_signal.emit(f"STATS ID {target_id} | captured: {stats_entry}")


                                    # SAVE TO STORAGE
                                    if stats_entry:
                                        print(f"DEBUG: Saving stats for ID {target_id}: {stats_entry}")
                                        self.inspection_results[target_id] = stats_entry

                                    self.inspection_substate = "INS_ALIGN_OBJ"
                                    self.sub_attempts = 0

                                    # Pause the worker state machine
                                    self.scan_state = self.WAIT_FOR_LLM
                                    self.request_llm_analysis_signal.emit(defect_data)

                elif self.scan_state == "RESETTING":
                    # Step 1: Return Linear Actuator (ID 2) to 0
                    if abs(self.current_cam_height) > 0.1:
                        # Move negative of current position to get back to 0
                        self.request_motor_move.emit(self.actuator_id, float(-self.current_cam_height), 1.0)
                        
                        # Set tracker to 0 immediately so we don't send command again in next loop
                        self.current_cam_height = 0.0 
                        self.last_command_time = now # Reset timer to wait for move to finish
                    
                    # Step 2: Return Turntable (ID 1) to 0
                    # We wait 2.0 seconds after the linear move starts before moving the table
                    elif abs(self.current_angle) > 0.1:
                        if now - self.last_command_time > 2.0:
                            self.request_motor_move.emit(1, float(-self.current_angle), 1.0)
                            
                            # Set tracker to 0 immediately
                            self.current_angle = 0.0
                            self.last_command_time = now # Reset timer
                            
                    # Step 3: Finish
                    else:
                        # Wait a moment for final moves to settle
                        if now - self.last_command_time > 2.0:
                            self.system_log_signal.emit("System Reset Complete. Ready.")
                            self.scan_state = "IDLE"

                # =================================================
                # TILE 3: SEGMENTATION
                # =================================================
                seg_vis = raw_bgr.copy() 
                if self.seg_model:
                    results = self.seg_model.predict(seg_vis, verbose=False, retina_masks=True)
                    result = results[0]
                    overlay = np.zeros_like(seg_vis)
                    class_areas = {0: 0.0, 1: 0.0, 2: 0.0}

                    if result.masks is not None:
                        for seg, cls_id in zip(result.masks.xy, result.boxes.cls):
                            cls_idx = int(cls_id)
                            color = self.seg_colors[cls_idx % len(self.seg_colors)]
                            poly = np.array(seg, dtype=np.int32)
                            cv2.fillPoly(overlay, [poly], color)
                            area = cv2.contourArea(poly)
                            class_areas[cls_idx] = class_areas.get(cls_idx, 0) + area

                        seg_vis = cv2.addWeighted(seg_vis, 1.0, overlay, 0.4, 0)
                        
                        mat_plastic = class_areas.get(0, 0) + class_areas.get(1, 0)
                        mat_copper = class_areas.get(1, 0) + class_areas.get(2, 0)
                        total_mat = mat_plastic + mat_copper
                        stats = {
                            "plastic_pct": (mat_plastic / total_mat) * 100 if total_mat > 0 else 0,
                            "copper_pct": (mat_copper / total_mat) * 100 if total_mat > 0 else 0
                        }
                        self.update_stats_signal.emit(stats)

                final_rgb = cv2.cvtColor(seg_vis, cv2.COLOR_BGR2RGB)
                self.change_rgb_signal.emit(QImage(final_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888))

                # =================================================
                # TILE 4: DEFECT DETECTION
                # =================================================
                def_vis = raw_bgr.copy()
                if self.def_model:
                    results = self.def_model.predict(def_vis, verbose=False, conf=0.4)
                    for result in results:
                        if result.boxes:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = f"{result.names[cls_id]}: {conf:.2f}"
                                cv2.rectangle(def_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(def_vis, (x1, y1 - 20), (x1 + w_text, y1), (0, 0, 255), -1)
                                cv2.putText(def_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                def_rgb = cv2.cvtColor(def_vis, cv2.COLOR_BGR2RGB)
                self.change_defect_signal.emit(QImage(def_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888))

                # =================================================
                # TILE 5: DEPTH (CROPPED BY OBJECT BOX)
                # =================================================
                self.threshold_filter.set_option(rs.option.min_distance, 0.12)
                self.threshold_filter.set_option(rs.option.max_distance, 0.3)
                filtered_depth_frame = self.threshold_filter.process(depth_frame)
                
                # Colorize the depth first
                depth_colormap = np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data())
                
                # Apply Bounding Box Mask to the visualization
                res_obj = self.class_model(raw_bgr, verbose=False)
                if len(res_obj[0].boxes) > 0:
                    # Find largest box
                    b = max(res_obj[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
                    dx1, dy1, dx2, dy2 = b.xyxy[0].cpu().numpy().astype(int)
                    
                    # Create a black mask and copy only the object area
                    final_depth_vis = np.zeros_like(depth_colormap)
                    final_depth_vis[dy1:dy2, dx1:dx2] = depth_colormap[dy1:dy2, dx1:dx2]
                else:
                    # If no object detected, show nothing or the raw filtered depth
                    final_depth_vis = np.zeros_like(depth_colormap)

                h_d, w_d, ch_d = final_depth_vis.shape
                self.change_depth_signal.emit(QImage(final_depth_vis.data, w_d, h_d, ch_d * w_d, QImage.Format.Format_RGB888))

            except Exception as e:
                print(f"Loop Error: {e}")
                break
        self.pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()