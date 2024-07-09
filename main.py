import streamlit as st
import cv2
import numpy as np
import pytesseract
from solver import solve_sudoku

# Adjust Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to find largest contour
def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# Function to extract grid
def extract_grid(image, contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warp
    else:
        raise Exception("Grid detection failed")

# Function to segment cells
def segment_cells(grid):
    cell_images = []
    cell_height, cell_width = grid.shape[0] // 9, grid.shape[1] // 9
    for i in range(9):
        row = []
        for j in range(9):
            cell = grid[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            row.append(cell)
        cell_images.append(row)
    return cell_images

# Function to recognize digit using Tesseract OCR
def recognize_digit_tesseract(cell_image):
    cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    cell_image = cv2.resize(cell_image, (28, 28))
    cell_image = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(cell_image, config='--psm 10 digits')
    digit = text.strip()
    if digit.isdigit() and 1 <= int(digit) <= 9:
        return int(digit)
    else:
        return 0

# Function to create grid matrix
def create_grid_matrix(cell_images):
    grid_matrix = []
    for row in cell_images:
        row_digits = []
        for cell in row:
            digit = recognize_digit_tesseract(cell)
            row_digits.append(digit)
        grid_matrix.append(row_digits)
    return grid_matrix

# Streamlit app code
def main():
    st.title('Sudoku Solver')

    # Upload image file
    uploaded_file = st.file_uploader("Upload a Sudoku Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

        # Process Sudoku image
        thresh = preprocess_image(image)
        largest_contour = find_largest_contour(thresh)
        grid = extract_grid(image, largest_contour)
        cell_images = segment_cells(grid)
        grid_matrix = create_grid_matrix(cell_images)

        # Display extracted Sudoku grid
        st.header("Extracted Sudoku Grid")
        st.write(np.matrix(grid_matrix))

        # Solve Sudoku
        solve_sudoku(grid_matrix)

        # Display solved Sudoku grid
        st.header("Solved Sudoku Grid")
        st.write(np.matrix(grid_matrix))

if __name__ == '__main__':
    main()
