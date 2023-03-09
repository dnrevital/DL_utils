# This code get two distorted images with it yml's files and output scaled, cropped & stitched image
#                                   Wrote: Eyal Naimy

#                                   Boundaries Indexes
#                  ------------------------------------------------
#                  |                                              |
#                  |        * - a                       * - b     |
#                  |                                              |
#                  |                                              |
#                  |        * - c                       * - d     |
#                  |                                              |
#                  ------------------------------------------------

#                                     Sides Indexes
#                                          Up
#                  ------------------------------------------------
#                  |                                              |
#                  |                                              |
#             Left |                                              | Right
#                  |                                              |
#                  |                                              |
#                  |                                              |
#                  ------------------------------------------------
#                                          Down


import cam_cal_scale_crop as crop
import cam_cal_stitch as st
import stitch_params as sp
import s_c_params as sc
import cv2 as cv
import matplotlib.pyplot as plt


def get_scaled_cropped(img, yml_filename):
    # load undistort params
    image_size, ret, mtx, dist = crop.load_cam_calibration_params_yaml(yml_filename)

    # undistortion operation
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (int(image_size[0]), int(image_size[1])), 0.4, (w, h))
    undist_crop = cv.undistort(img, mtx, dist, None, newcameramtx)

    # scaling - calc
    conf_x, conf_y, point_a, point_b, point_c, point_d, resize_x, resize_y = crop.scale(undist_crop, sc.NX)

    # boundaries cropping - calc
    if conf_x > 0.9 and conf_y > 0.9:
        cr_max_x, cr_min_x, cr_max_y, cr_min_y = crop.crop_points(undist_crop, point_a, point_b, point_c, point_d,
                                                                  sc.NX, sc.LAYERS_GAP, resize_x, resize_y)

        # scale - implement
        un_scaled = cv.resize(undist_crop, (0, 0), fx=resize_x, fy=resize_y)

        # cropping - implement
        un_res_crop = un_scaled[int(cr_min_y):int(cr_max_y), int(cr_min_x):int(cr_max_x)]

        return un_res_crop

    return 0


def stitch_proc():

    main_cam_edge = sp.SIDE_CAM_D       # see: stitch_params.py
    sec_cam_edge = sp.SIDE_CAM_U

    # load image-a and perform cropping & scaling
    img_a = cv.imread('CAM22.jpg')
    yml_filename_a = 'cal_params_CAM22_V4_12X14 - Best.yml'
    sc_img_a = get_scaled_cropped(img_a, yml_filename_a)

    # load image-b and perform cropping & scaling
    img_b = cv.imread('CAM24.jpg')
    yml_filename_b = 'cal_params_CAM24_V1_12X14 - Best.yml'
    sc_img_b = get_scaled_cropped(img_b, yml_filename_b)

    # expose images
    plt.imshow(img_a)
    plt.title("Original-A")
    plt.show()     
    
    plt.imshow(sc_img_a)
    plt.title("undist and scaled-A")
    plt.show()     

    plt.imshow(img_b)
    plt.title("Original-B")
    plt.show()     

    plt.imshow(sc_img_b)
    plt.title("undist and scaled-B")
    plt.show()     

    # stitch process
    st.stitch_proc(sc_img_a, sc_img_b, sc.NX, main_cam_edge, sec_cam_edge)


if __name__ == '__main__':
    stitch_proc()
