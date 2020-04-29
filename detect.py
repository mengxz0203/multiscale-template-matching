import cv2 as cv
import numpy as np


def template_demo():
    template = cv.imread('weibo_ipad.png', 1)
    target = cv.imread('images/1.jpg', 1)
    # target = cv.pyrDown(target)
    # target = cv.pyrDown(target)
    cv.imshow('template image', template)
    cv.imshow('target image', target)
    # cv.imwrite('images/test.jpg', target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED]
    th, tw = template.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, template, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow('match' + np.str(md), target)


template_demo()
cv.waitKey(0)
cv.destroyAllWindows()
