import cv2
import numpy as np
import copy
import tensorflow as tf
import os

# Load the trained digit recognition model
try:
    if not os.path.exists("digit_model.keras"):
        raise FileNotFoundError("digit_model.keras not found")
    DIGIT_MODEL = tf.keras.models.load_model("digit_model.keras")
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"Error: Could not load digit model: {e}")
    print("Please run 'python train_digit_model.py' first to train the model.")
    MODEL_AVAILABLE = False
    DIGIT_MODEL = None

# SOLVER LOGIC

def find_empty(bo):
    for i in range(9):
        for j in range(9):
            if bo[i][j] == 0:
                return (i, j)
    return False

def valid(bo, num, pos):
    # row
    if num in bo[pos[0]]:
        return False
    # col
    col_vals = [bo[i][pos[1]] for i in range(9)]
    if num in col_vals:
        return False
    # box
    rs = (pos[0] // 3) * 3
    cs = (pos[1] // 3) * 3
    for r in range(rs, rs + 3):
        for c in range(cs, cs + 3):
            if bo[r][c] == num:
                return False
    return True

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    r, c = find
    for v in range(1, 10):
        if valid(bo, v, (r, c)):
            bo[r][c] = v
            if solve(bo):
                return True
            bo[r][c] = 0
    return False

def print_board(bo):
    temp = copy.deepcopy(bo)
    print()
    for i in range(9):
        for j in range(9):
            if temp[i][j] == 0:
                temp[i][j] = "X"
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - -")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 0:
                print(" ", end="")
            if j == 8:
                print(temp[i][j])
            else:
                print(str(temp[i][j]) + " ", end="")

def cleanup_ocr_board(board):
    cleaned = copy.deepcopy(board)
    for r in range(9):
        for c in range(9):
            v = cleaned[r][c]
            if v != 0:
                cleaned[r][c] = 0
                if valid(cleaned, v, (r, c)):
                    cleaned[r][c] = v
                else:
                    cleaned[r][c] = 0
    return cleaned

# WEBCAM HELPERS

def capture_frame_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Press 'c' to capture Sudoku image, 'q' to quit.")
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Sudoku Capture (press 'c' to capture, 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            captured = frame.copy()
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top left
    rect[2] = pts[np.argmax(s)]   # bottom right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top right
    rect[3] = pts[np.argmax(diff)]  # bottom left
    return rect

def find_sudoku_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            best = approx
    if best is None:
        return None, None
    pts = best.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    return rect, thr

def warp_sudoku(image, rect, size=900):
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped, M, Minv, size

# CNN for digit recognition

def prepare_cell_for_model(cell_img):
    # grayscale cell, cropped to central area
    if cell_img.size == 0:
        return None
    blur = cv2.GaussianBlur(cell_img, (3, 3), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = thresh.shape
    if cv2.countNonZero(thresh) < 0.06 * h * w:
        return None

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(cnt)
    if cw * ch < 0.04 * h * w:
        return None

    digit = thresh[y:y+ch, x:x+cw]

    side = max(cw, ch)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off:y_off+ch, x_off:x_off+cw] = digit

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0

def recognize_digit(cell_img):
    if not MODEL_AVAILABLE or DIGIT_MODEL is None:
        print("Error: Digit model not available.")
        return 0
    arr = prepare_cell_for_model(cell_img)
    if arr is None:
        return 0
    arr = arr[np.newaxis, ..., np.newaxis]
    preds = DIGIT_MODEL.predict(arr, verbose=0)[0]
    digit = int(np.argmax(preds))
    conf = float(preds[digit])
    if digit == 0 or conf < 0.80:
        return 0
    return digit

def extract_grid(warped, size=900):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    step_y = h // 9
    step_x = w // 9
    board = [[0 for _ in range(9)] for _ in range(9)]

    for r in range(9):
        for c in range(9):
            y1 = r * step_y
            y2 = (r + 1) * step_y
            x1 = c * step_x
            x2 = (c + 1) * step_x

            cell = gray[y1:y2, x1:x2]
            # strong center crop to avoid thick lines
            margin_y = int(step_y * 0.2)
            margin_x = int(step_x * 0.2)
            cell = cell[margin_y:step_y - margin_y,
                        margin_x:step_x - margin_x]

            digit = recognize_digit(cell)
            board[r][c] = digit
    return board

def overlay_solution_on_image(orig_image, Minv, original_board, solved_board, size=900):
    result = orig_image.copy()
    cell_size = size // 9
    for r in range(9):
        for c in range(9):
            if original_board[r][c] == 0 and solved_board[r][c] != 0:
                x = (c + 0.5) * cell_size
                y = (r + 0.5) * cell_size
                pts = np.array([[[x, y]]], dtype="float32")
                dst = cv2.perspectiveTransform(pts, Minv)
                px, py = dst[0][0]
                cv2.putText(
                    result,
                    str(solved_board[r][c]),
                    (int(px - cell_size * 0.3), int(py + cell_size * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
    return result


def main():
    frame = capture_frame_from_webcam()
    if frame is None:
        print("No image captured.")
        return

    rect, _ = find_sudoku_contour(frame)
    if rect is None:
        print("Could not find a Sudoku grid in the image.")
        return

    warped, M, Minv, size = warp_sudoku(frame, rect, size=900)
    board = extract_grid(warped, size=size)
    board = cleanup_ocr_board(board)

    print("Detected puzzle after cleanup (0 = empty):")
    print_board(board)

    original_board = copy.deepcopy(board)

    if not solve(board):
        print("\nNo solution found.")
        return

    print("\nSolved Sudoku:")
    print_board(board)

    solved_img = overlay_solution_on_image(frame, Minv, original_board, board, size=size)
    cv2.imshow("Solved Sudoku", solved_img)
    print("\nPress any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
