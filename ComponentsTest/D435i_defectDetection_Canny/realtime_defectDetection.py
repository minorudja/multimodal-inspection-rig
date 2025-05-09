import pyrealsense2 as rs
import numpy as np
import cv2

# Setup RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

print("Detecting surface fractures (holes, dents). Press ESC to exit.")

def detect_fractures(gray_img):
    """Detect small dark enclosed contours indicating fractures."""
    # Optional: Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_img)

    # Threshold to highlight dark regions (potential fractures)
    _, thresh = cv2.threshold(gray_eq, 68, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours of dark regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fractures = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 65 < area < 1000:  # Heuristic: ignore too small or large regions
            x, y, w, h = cv2.boundingRect(cnt)
            fractures.append((x, y, w, h))

    return fractures, thresh

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect fractures
        fracture_boxes, threshold_map = detect_fractures(gray)

        # Draw detected fracture boxes
        output = color_image.copy()
        for (x, y, w, h) in fracture_boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, "Fracture", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        # Show original + threshold side-by-side
        thresh_bgr = cv2.cvtColor(threshold_map, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((output, thresh_bgr))
        cv2.imshow("Fracture Detection | Output (left) + Threshold (right)", combined)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()