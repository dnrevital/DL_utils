# This code get array of undistorted images and output combined image

#        Cameras Mapping Marking Format
#               ***_B_1
#              A---B---C---D---E
#           0
#           1      *
#           2
#           3
#           4


from dataclasses import dataclass
import stitch_params as sp
from scipy import ndimage
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import math


@dataclass
class Point:
    image: float
    fname: str


def upload_images(images_path):
    images = []
    image_path_list = glob.glob(images_path)
    for fname in image_path_list:
        img = cv.imread(fname)
        p = Point(img, fname)
        images.append(p)

    return images


def crop_roi(image, cam_side, h, w):
    crop_1_im = []
    crop_2_im = []
    if cam_side == sp.SIDE_CAM_U:
        crop_1_im = image[0:int(h / 2), 0:int(w / 2)]    # crop - a
        crop_2_im = image[0:int(h / 2), int(w / 2):w]    # crop - b
    elif cam_side == sp.SIDE_CAM_D:
        crop_1_im = image[int(h / 2):h, 0:int(w / 2)]    # crop - c
        crop_2_im = image[int(h / 2):h, int(w / 2):w]    # crop - d
    elif cam_side == sp.SIDE_CAM_L:
        crop_1_im = image[0:int(h / 2), 0:int(w / 2)]    # crop - a
        crop_2_im = image[int(h / 2):h, 0:int(w / 2)]    # crop - c
    elif cam_side == sp.SIDE_CAM_R:
        crop_1_im = image[0:int(h / 2), int(w / 2):w]    # crop - b
        crop_2_im = image[int(h / 2):h, int(w / 2):w]    # crop - d

    return crop_1_im, crop_2_im


def quarter_coords_corection(pnts, cam_side, h, w):
    # add cropping quarter
    if cam_side == sp.SIDE_CAM_U:                        # point a & b
        pnts[1][0] = pnts[1][0] + float(w / 2)           # point - b
    elif cam_side == sp.SIDE_CAM_D:                      # point c & d
        pnts[0][1] = pnts[0][1] + float(h / 2)           # point - c
        pnts[1][0] = pnts[1][0] + float(w / 2)           # point - d
        pnts[1][1] = pnts[1][1] + float(h / 2)           # point - d
    elif cam_side == sp.SIDE_CAM_L:                      # point a & c
        pnts[1][1] = pnts[1][1] + float(h / 2)           # point - c
    elif cam_side == sp.SIDE_CAM_R:                      # point b & d
        pnts[0][0] = pnts[0][0] + float(w / 2)           # point - b
        pnts[1][1] = pnts[1][1] + float(h / 2)           # point - d
        pnts[1][0] = pnts[1][0] + float(w / 2)           # point - d

    return pnts


def restore_imgb_coords(pnts, cam_side, h, w):
    # add cropping quarter
    if cam_side == sp.SIDE_CAM_U:  # point a & b
        pnts[1][0] = pnts[1][0] + float(w / 2)  # point - b
    elif cam_side == sp.SIDE_CAM_L:  # point a & c
        pnts[1][1] = pnts[1][1] + float(h / 2)  # point - c
    elif cam_side == sp.SIDE_CAM_R:  # point b & d
        pnts[1][0] = pnts[1][0] + float(w / 2)  # point - b
        pnts[1][1] = pnts[1][1] + float(h / 2)  # point - d
        pnts[1][0] = pnts[1][0] + float(w / 2)  # point - d

    return pnts


def stitch_proc(image_cam_a, image_cam_b, nx, cam_side_a, cam_side_b):

    # expose images
    plt.imshow(image_cam_a)
    plt.title("image_cam_a")
    plt.show()     

    plt.imshow(image_cam_b)
    plt.title("image_cam_b")
    plt.show()     

    # adaptive histogram for better target detection
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # find boundaries points
    # image_a
    image_a_gl = cv.cvtColor(image_cam_a, cv.COLOR_BGR2GRAY)
    image_a_gl = clahe.apply(image_a_gl)
    h_a, w_a = image_a_gl.shape[:2]
    crop_1_im_a, crop_2_im_a = crop_roi(image_a_gl, cam_side_a, h_a, w_a)
    ret, point_c_im_a = cv.findChessboardCorners(crop_1_im_a, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    ret, point_d_im_a = cv.findChessboardCorners(crop_2_im_a, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    # image_b
    image_b_gl = cv.cvtColor(image_cam_b, cv.COLOR_BGR2GRAY)
    image_b_gl = clahe.apply(image_b_gl)
    h_b, w_b = image_b_gl.shape[:2]
    crop_1_im_b, crop_2_im_b = crop_roi(image_b_gl, cam_side_b, h_b, w_b)
    ret, point_a_im_b = cv.findChessboardCorners(crop_1_im_b, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    ret, point_b_im_b = cv.findChessboardCorners(crop_2_im_b, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    # p_a & p_b are the coordinates points for calculating the roll angle between img_a & img_b
    p_a = [point_c_im_a[int((nx*nx) / 2)][0, :], point_d_im_a[int((nx*nx) / 2)][0, :]]
    p_b = [point_a_im_b[int((nx*nx) / 2)][0, :], point_b_im_b[int((nx*nx) / 2)][0, :]]

    # quarter coordinates in global pose
    p_a = quarter_coords_corection(p_a, cam_side_a, h_a, w_a)
    p_b = quarter_coords_corection(p_b, cam_side_b, h_b, w_b)

    # calculate images angles
    tan_a = 0
    tan_b = 0
    ang_coeff = -1
    if cam_side_a == sp.SIDE_CAM_U or cam_side_a == sp.SIDE_CAM_D:      # U-D sides
        tan_a = (p_a[0][1] - p_a[1][1]) / (p_a[0][0] - p_a[1][0])
        tan_b = (p_b[0][1] - p_b[1][1]) / (p_b[0][0] - p_b[1][0])
    elif cam_side_a == sp.SIDE_CAM_L or cam_side_a == sp.SIDE_CAM_R:    # R-L sides
        tan_a = (p_a[0][0]-p_a[1][0]) / (p_a[0][1]-p_a[1][1])
        tan_b = (p_b[0][0]-p_b[1][0]) / (p_b[0][1]-p_b[1][1])
        ang_coeff = 1

    deg_a = math.atan(tan_a)
    deg_b = math.atan(tan_b)

    rot_angle = ang_coeff * math.degrees(deg_a - deg_b)

    # align roll images
    image_b_rot = ndimage.rotate(image_cam_b, rot_angle, reshape=True)
    plt.imshow(image_b_rot)
    plt.title("image_b_rot")
    plt.show()     

    #       calculate boundaries after rolling
    # image_b
    # clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    image_b_rot_gl = cv.cvtColor(image_b_rot, cv.COLOR_BGR2GRAY)
    h, w = image_b_rot_gl.shape[:2]
    crop_a_im_b_rot, crop_b_im_b_rot = crop_roi(image_b_rot_gl, cam_side_b, h, w)

    # find corners
    ret, point_a_im_b_rot = cv.findChessboardCorners(crop_a_im_b_rot, (nx, nx),
                                                     flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret:
        ret, point_a_im_b_rot = cv.findChessboardCorners(clahe.apply(crop_a_im_b_rot), (nx, nx),
                                                         flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret, point_b_im_b_rot = cv.findChessboardCorners(crop_b_im_b_rot, (nx, nx),
                                                     flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret:
        ret, point_b_im_b_rot = cv.findChessboardCorners(clahe.apply(crop_b_im_b_rot), (nx, nx),
                                                         flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    p_b_rot = [point_a_im_b_rot[int((nx*nx) / 2)][0, :], point_b_im_b_rot[int((nx*nx) / 2)][0, :]]
    p_b_rot = restore_imgb_coords(p_b_rot, cam_side_b, h, w)

    # shift offset
    diff_px = [p_a[0][0] - p_b_rot[0][0], p_a[1][0] - p_b_rot[1][0]]
    diff_py = [p_a[0][1] - p_b_rot[0][1], p_a[1][1] - p_b_rot[1][1]]

    # averaging offset
    av_dx = int(sum(diff_px) / len(diff_px))
    av_dy = int(sum(diff_py) / len(diff_py))

    # shift
    h_a, w_a = image_cam_a.shape[:2]
    h_b, w_b = image_b_rot.shape[:2]
    if w_a > w_b:
        w_g = w_a
    else:
        w_g = w_b
    if h_a > h_b:
        h_g = h_a
    else:
        h_g = h_b

    master = np.zeros([h_g + abs(av_dy), w_g + abs(av_dx), 3], dtype=np.uint8)
    master.fill(0)

    alpha = 0.5
    if av_dx < 0 and av_dy < 0:
        master[abs(av_dy):h_a+abs(av_dy), abs(av_dx):w_a+abs(av_dx)] = image_cam_a
        master[0:h_b, 0:w_b] = image_b_rot
    elif av_dx < 0 < av_dy:
        master[0:h_a, abs(av_dx):w_a+abs(av_dx)] = image_cam_a*alpha
        master[av_dy:h_b+av_dy, 0:w_b] = (image_b_rot*alpha) + (master[av_dy:h_b+av_dy, 0:w_b]*alpha)
    elif av_dx > 0 > av_dy:
        master[abs(av_dy):h_a+abs(av_dy), av_dx:w_a+av_dx] = image_cam_a*alpha
        master[abs(av_dy):h_b+abs(av_dy), 0:w_b] = image_b_rot + (master[abs(av_dy):h_b+abs(av_dy), 0:w_b]*alpha)
    else:
        master[0:h_a, 0:w_a] = image_cam_a*alpha
        master[av_dy:h_b+av_dy, av_dx:w_b+av_dx] = image_b_rot*alpha + (master[av_dy:h_b+av_dy, av_dx:w_b+av_dx]*alpha)

    plt.imshow(master)
    plt.title("master")
    plt.show()     
 