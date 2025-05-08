import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Create output folder
output_folder = "captured_images"
os.makedirs(output_folder, exist_ok=True)

img_counter = 1

print("Press SPACEBAR to capture RGB + Depth. Press ESC to exit.")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth to colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Display combined image
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense - SPACE to Capture | ESC to Exit", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Exiting...")
            break
        elif key == 32:  # SPACEBAR
            # Define filenames
            base_name = f"imgCapture_{img_counter}"
            rgb_path = os.path.join(output_folder, f"{base_name}_color.png")
            depth_raw_path = os.path.join(output_folder, f"{base_name}_depth.png")
            depth_colormap_path = os.path.join(output_folder, f"{base_name}_depth_colormap.png")

            # Save images
            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_raw_path, depth_image)  # 16-bit raw depth
            cv2.imwrite(depth_colormap_path, depth_colormap)

            print(f"Captured: {rgb_path}, {depth_raw_path}, {depth_colormap_path}")
            img_counter += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()