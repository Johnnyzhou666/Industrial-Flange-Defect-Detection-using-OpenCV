import os
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def generate_test_sample(filepath="test_flange_01.png"):
    """
    Generates a synthetic flange image with simulated defects for testing.
    Saved as .png to prevent JPEG lossy compression artifacts during HSV masking.
    """
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Base structure
    center = (400, 300)
    cv2.rectangle(img, (150, 240), (650, 360), (170, 170, 170), -1)
    cv2.circle(img, center, 160, (170, 170, 170), -1)
    
    # Through holes
    cv2.circle(img, center, 60, (255, 255, 255), -1)
    cv2.circle(img, (200, 300), 25, (255, 255, 255), -1)
    cv2.circle(img, (600, 300), 25, (255, 255, 255), -1)
    
    # Contours/shadows for depth
    cv2.circle(img, center, 160, (140, 140, 140), 3)
    cv2.circle(img, center, 60, (140, 140, 140), 2)
    cv2.rectangle(img, (150, 240), (650, 360), (140, 140, 140), 2)

    # Inject defects
    # 1. Rust/Stain
    rust_pts = np.array([[360, 180], [400, 170], [420, 190], [390, 210], [350, 200]], np.int32)
    cv2.fillPoly(img, [rust_pts], (60, 80, 100))
    
    # 2. Deep scratch
    scratch_pts = np.array([[460, 380], [480, 395], [490, 390], [520, 420]])
    cv2.polylines(img, [scratch_pts], isClosed=False, color=(40, 40, 40), thickness=4)
    
    # 3. Chipped edge
    chip_pts = np.array([[270, 140], [290, 160], [310, 135], [290, 110]], np.int32)
    cv2.fillPoly(img, [chip_pts], (255, 255, 255))
    
    # 4. Background noise (High density, pure dark pixels to test robustness)
    mask = np.random.rand(600, 800) < 0.015 
    part_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 200 
    combined_mask = mask & part_mask
    img[combined_mask] = [30, 30, 30] 
    
    cv2.imwrite(filepath, img)
    return filepath

def run_inspection(image_path):
    print(f"[INFO] Initializing inspection for: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to load image at {image_path}")
        return

    output_img = img.copy()
    status_flags = []

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dimensional & Edge Check
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Spec limits for the flange part
        TARGET_W, TARGET_H = 500, 320 
        TOLERANCE = 15 
        
        if not (TARGET_W - TOLERANCE <= w <= TARGET_W + TOLERANCE) or \
           not (TARGET_H - TOLERANCE <= h <= TARGET_H + TOLERANCE):
            status_flags.append("DIMENSION_FAIL")
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_img, f"W:{w} H:{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Surface Defect Check (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 130]) # Relaxed threshold to prevent false negatives
    
    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Use CLOSE to maintain scratch continuity
    defect_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for dc in defect_contours:
        # Lowered threshold (>10) to catch segmented scratches, while still ignoring 1-pixel noise
        if cv2.contourArea(dc) > 10: 
            if "SURFACE_DEFECT" not in status_flags:
                status_flags.append("SURFACE_DEFECT")
            dx, dy, dw, dh = cv2.boundingRect(dc)
            cv2.rectangle(output_img, (dx, dy), (dx+dw, dy+dh), (0, 0, 255), 2) 
            cv2.putText(output_img, "DEFECT", (dx, dy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Final Evaluation
    is_pass = len(status_flags) == 0
    final_status = "PASS" if is_pass else "FAIL: " + " | ".join(status_flags)
    color = (0, 255, 0) if is_pass else (0, 0, 255)
    
    cv2.putText(output_img, final_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    print("[INFO] Original Sample (With enhanced noise, scratches, and chipped edge):")
    cv2_imshow(img)
    print(f"\n[RESULT] Inspection Outcome: {final_status}")
    cv2_imshow(output_img)

if __name__ == "__main__":
    # Using .png format to ensure zero color drift during inspection
    sample_file = "test_flange_01.png"
    
    generate_test_sample(sample_file)
    run_inspection(sample_file)
    
    # Cleanup generated file
    if os.path.exists(sample_file):
        os.remove(sample_file)
