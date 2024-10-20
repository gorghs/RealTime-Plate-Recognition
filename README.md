# Number Plate Recognition

This project implements a real-time number plate recognition system using OpenCV and Tesseract OCR. The system captures video from a camera, detects number plates in the frames, and extracts the text from those plates.

## Features

- Real-time number plate detection and recognition.
- Efficient preprocessing steps to improve OCR accuracy.
- Saves recognized number plates to an Excel file.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas
- Pillow
- Pytesseract
- Tesseract OCR

## Installation

1. Clone the repository:

    ```bash
    git clone [https://github.com/karthick-V-212223040086/RealTime-Plate-Recognition.git]
    cd number-plate-recognition
    ```

2. Install the required packages:

    ```bash
    pip install opencv-python numpy pandas Pillow pytesseract
    ```

## Install Tesseract OCR:

Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).

Set the Tesseract executable path in the script if necessary (uncomment the line in the code).

## Usage

Run the script:

   ```bash
   python number_plate_recognition.py
The camera feed will start, and the system will begin recognizing number plates. Press 'q' to quit the application.

Recognized number plates will be saved to an Excel file named `recognized_number_plates.xlsx` in the current directory.

## Code Explanation

The main components of the script are:

- **Number Plate Detection**: Using contour detection and morphological operations to identify and isolate potential number plates.
- **Image Preprocessing**: Techniques such as Gaussian blur and adaptive thresholding are used to enhance the image for better OCR results.
- **OCR Processing**: Pytesseract is employed to recognize text from the processed images of number plates.
- **Results Handling**: The script avoids duplicate entries and saves recognized plates in an Excel file.

## Contributing

Feel free to contribute by submitting issues, feature requests, or pull requests!

## Acknowledgements

- OpenCV
- Tesseract OCR

