import cv2
import numpy as np



# img = cv2.imread('../examples/Adirondack-perfect/im0.png')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
#
#
# kp = sift.detect(gray,None)#找到关键点
#
# img=cv2.drawKeypoints(gray,kp,img)#绘制关键点
#
# cv2.imshow('sp',img)
# cv2.waitKey(0)


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)

if __name__ == '__main__':
    img1_gray = cv2.imread("../examples/std/lena.jpg")
    img2_gray = cv2.imread("../examples/std/lena.jpg")

    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SURF()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)

    drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:100])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
