import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

class ImgSplit():
    def __init__(self):
        rospy.init_node('img_split')
        rospy.Subscriber('/usb_cam/image_raw', Image, self.callback_cam_1, queue_size = 10)
        rospy.Subscriber('/usb_cam1/image_raw', Image, self.callback_cam_2, queue_size = 10)
        self.cvimg_1 = ''
        self.cvimg_2 = ''
        # self.final_img = ''
        self.pub = rospy.Publisher('/image_final', Image, queue_size = 10)
        

    def callback_cam_1(self, msg):
        bridge = CvBridge()
        self.cvimg_1 = bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        rospy.loginfo('convert successfully.')

    def callback_cam_2(self, msg):
        bridge = CvBridge()
        self.cvimg_2 = bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        rospy.loginfo('convert successfully.')
        if len(self.cvimg_1) > 0:
            # print('```````````')
            # self.split()
            self.img_plus()

    def img_plus(self):
        plus_img = np.hstack((self.cvimg_1, self.cvimg_2))
        brg = CvBridge()
        final_img = brg.cv2_to_imgmsg(plus_img,'bgr8')
        
        self.pub.publish(final_img)

    def split(self):
        MIN = 10
        starttime = time.time()
        img1 = self.cvimg_1
        img2 = self.cvimg_2
        surf = cv2.xfeatures2d.SURF_create(10000, nOctaves=4, extended=False, upright=True)
        kp1, descrip1 = surf.detectAndCompute(img1, None)
        kp2, descrip2 = surf.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)

        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        match = flann.knnMatch(descrip1, descrip2, k=2)

        good = []
        for i, (m, n) in enumerate(match):
            if (m.distance < 0.75 * n.distance):
                good.append(m)

        if len(good) > MIN:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
            warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))
            direct = warpImg.copy()
            direct[0:img1.shape[0], 0:img1.shape[1]] = img1
            simple = time.time()
            rows, cols = img1.shape[:2]
            for col in range(0, cols):
                if img1[:, col].any() and warpImg[:, col].any():  # start chongdie zui zuo
                    left = col
                    break
            for col in range(cols - 1, 0, -1):
                if img1[:, col].any() and warpImg[:, col].any():  # chongdie zuiyou
                    right = col
                    break
            res = np.zeros([rows, cols, 3], np.uint8)
            for row in range(0, rows):
                for col in range(0, cols):
                    if not img1[row, col].any():  # meiyou yuantu ,xuanzhuan tianchong
                        res[row, col] = warpImg[row, col]
                    elif not warpImg[row, col].any():
                        res[row, col] = img1[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
            warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
            final = time.time()
            img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
            brg = CvBridge()
            final_img = brg.cv2_to_imgmsg(img3,'bgr8')
            self.pub.publish(final_img)
            # plt.imshow(img3, ), plt.show()
            print("simple stich cost %f" % (simple - starttime))
            print("\ntotal cost %f" % (final - starttime))
        else:
            print("not enough matches!")

if __name__ == '__main__':
    ImgSplit()
    rate = rospy.Rate(10)
    rate.sleep()
    rospy.spin()
    

