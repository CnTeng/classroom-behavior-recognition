import cv2 as cv
import matplotlib.pyplot as plt


def draw_hist_image(channels, hist_name, image_name):
    plt.hist(channels[0].flatten(), 256, [0, 256])
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.savefig(hist_name, dpi=300)
    plt.cla()
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, image)
    cv.imwrite(image_name, image)


image = cv.imread("./origin.jpg")

ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)

channels_he = cv.split(ycrcb)
channels_clahe = cv.split(ycrcb)

plt.hist(channels_he[0].flatten(), 256, [0, 256])
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.savefig("origin_hist.jpg", dpi=300)
plt.cla()

cv.equalizeHist(channels_he[0], channels_he[0])
draw_hist_image(channels_he, "he_hist.jpg", "he.jpg")

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe.apply(channels_clahe[0], channels_clahe[0])
draw_hist_image(channels_clahe, "clahe_hist.jpg", "clahe.jpg")
