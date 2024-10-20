import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import re
import os

# Set up Tesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Detecting number plate
def number_plate_detection(img):
    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        # Improved adaptive threshold for cleaning
        thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour_area = [cv2.contourArea(c) for c in contours]
            max_cntr_index = np.argmax(contour_area)

            max_cnt = contours[max_cntr_index]
            max_cntArea = contour_area[max_cntr_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            if not ratioCheck(max_cntArea, w, h):
                return plate, None

            final_img = thresh[y:y + h, x:x + w]
            return final_img, [x, y, w, h]

        else:
            return plate, None

    def ratioCheck(area, width, height):
        # Improved ratio checks for number plate dimensions
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        if (area < 1063.62 or area > 73862.5) or (ratio < 2 or ratio > 6):
            return False
        return True

    def isMaxWhite(plate):
        # Checking if the plate has a predominantly white background
        avg = np.mean(plate)
        return avg >= 120

    def ratio_and_rotation(rect):
        (x, y), (width, height), rect_angle = rect

        if width > height:
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 15:
            return False

        if height == 0 or width == 0:
            return False

        area = height * width
        return ratioCheck(area, width, height)

    # Preprocess the image for edge detection
    img2 = cv2.GaussianBlur(img, (5, 5), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection for better contours
    edges = cv2.Canny(img2, 100, 200)

    # Morphological operations to close gaps
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = edges.copy()
    cv2.morphologyEx(src=edges, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)

    contours, _ = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]
            if isMaxWhite(plate_img):
                clean_plate, rect = clean2_plate(plate_img)
                if rect:
                    clean_plate = cv2.resize(clean_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    clean_plate = cv2.fastNlMeansDenoising(clean_plate, None, 30, 7, 21)

                    # Improved OCR configuration
                    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    try:
                        plate_im = Image.fromarray(clean_plate)
                        text = pytesseract.image_to_string(plate_im, config=config)
                        
                        # Filter out invalid results (ensure it matches a number plate format)
                        text = re.sub('[^A-Z0-9]', '', text).upper()  # Remove non-alphanumeric characters
                        if 5 <= len(text) <= 10:  # Typical number plate length
                            return text.strip()
                    except Exception as e:
                        print(f"Error recognizing text: {e}")
                        continue

    return None

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

recognized_plates = []

# Capture video
print("Starting camera feed. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    number_plate = number_plate_detection(frame)
    if number_plate:
        res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate))).upper()
        if res2 and res2 not in recognized_plates:  # Avoid duplicates
            recognized_plates.append(res2)
            print("Recognized Number Plate:", res2)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Save recognized plates to Excel
if recognized_plates:
    df = pd.DataFrame(recognized_plates, columns=["Number Plates"])
    output_file = os.path.join(os.getcwd(), "recognized_number_plates.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Recognized number plates saved to {output_file}")
else:
    print("No number plates recognized.")
