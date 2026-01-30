
import cv2
import numpy as np

# load grayscale image
img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)

# sobel
Dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# harris structure tensor components
A = Dx * Dx
B = Dx * Dy
C = Dy * Dy

# gaussian blur
A_p = cv2.GaussianBlur(A, (5, 5), 1)
B_p = cv2.GaussianBlur(B, (5, 5), 1)
C_p = cv2.GaussianBlur(C, (5, 5), 1)

# harris response R = det(M) - k*(trace(M))^2
k = 0.05
detM = A_p * C_p - (B_p ** 2)
traceM = A_p + C_p
R = detM - k * (traceM ** 2)

# normalizing
R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
R_norm = R_norm.astype(np.uint8)

# threshold corners
threshold = 0.01 * R.max()
corners = np.where(R > threshold)

corners_img = img.copy()
for y, x in zip(corners[0], corners[1]):
    cv2.circle(corners_img, (x, y), 2, (0, 0, 255), -1)

cv2.imwrite("corners.jpg", corners_img)

# eigenvalues and eigenvectors at each corner
eigen_img = img.copy()
scale = 8

for y, x in zip(corners[0], corners[1]):
    M = np.array([[A_p[y, x], B_p[y, x]],
                  [B_p[y, x], C_p[y, x]]], dtype=np.float32)

    vals, vecs = np.linalg.eig(M)

    # eigenvectors
    e1 = vecs[:, np.argmax(vals)]
    e2 = vecs[:, np.argmin(vals)]

    # draw arrows
    pt1 = (int(x), int(y))

    pt_e1 = (int(x + scale * e1[0]), int(y + scale * e1[1]))
    pt_e2 = (int(x + scale * e2[0]), int(y + scale * e2[1]))

    cv2.arrowedLine(eigen_img, pt1, pt_e1, (0, 0, 255), 1)
    cv2.arrowedLine(eigen_img, pt1, pt_e2, (255, 0, 0), 1)

cv2.imwrite("eigenvectors.jpg", eigen_img)

print("Completed.")