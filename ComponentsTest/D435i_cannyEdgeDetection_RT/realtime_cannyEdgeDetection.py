import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Initialize pipeline and configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Create output folder
output_folder = "captured_edges"
os.makedirs(output_folder, exist_ok=True)

img_counter = 1

print("Press SPACEBAR to capture all images. Press ESC to exit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # --- RGB Edge Detection ---
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray_eq = clahe.apply(gray_blur)
        # edges_rgb = cv2.Canny(gray_eq, 50, 150)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray_blur)
        edges_rgb = cv2.Canny(gray_eq, 30, 100)

        # --- Depth Edge Detection ---
        depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
        # depth_blur = cv2.GaussianBlur(depth_8u, (5, 5), 1.4)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # depth_eq = clahe.apply(depth_blur)
        # edges_depth = cv2.Canny(depth_eq, 50, 150)
        depth_blur = cv2.GaussianBlur(depth_8u, (3, 3), 0.8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        depth_eq = clahe.apply(depth_blur)
        edges_depth = cv2.Canny(depth_eq, 30, 100)

        # Stack edges for display
        combined = np.hstack((edges_rgb, edges_depth))
        cv2.imshow("Canny Edges | RGB (left) vs Depth (right)", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Exiting...")
            break

        elif key == 32:  # SPACEBAR
            base_name = f"imgCapture_{img_counter}"
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_color.png"), color_image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_depth.png"), depth_image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_edges_rgb.png"), edges_rgb)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_edges_depth.png"), edges_depth)

            print(f"Captured: {base_name}_*.png")
            img_counter += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()