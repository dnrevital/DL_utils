# This code get unscaled image and output scaled image

#                                   Boundaries Indexes
#                  ------------------------------------------------
#                  |                                              |
#                  |        * - a                       * - b     |
#                  |                                              |
#                  |                                              |
#                  |        * - c                       * - d     |
#                  |                                              |
#                  ------------------------------------------------


import s_c_params
import cv2 as cv
import sys


def calc_aspect_ratio(point_a, point_b, point_c, point_d, nx):  # pixel to c"m ratio
    # point - A
    dx_x = abs(point_a[0][0, 0] - point_a[nx - 1][0, 0])/(nx-1)
    dy_x = abs(point_a[0][0, 1] - point_a[nx - 1][0, 1])/(nx-1)
    xss_a = (dx_x ** 2 + dy_x ** 2) ** 0.5    # x-Size target Square size (pixels)

    dx_y = abs(point_a[0][0, 0] - point_a[(nx*nx) - nx][0, 0]) / (nx-1)
    dy_y = abs(point_a[0][0, 1] - point_a[(nx*nx) - nx][0, 1]) / (nx-1)
    yss_a = (dx_y ** 2 + dy_y ** 2) ** 0.5    # y-Size target Square size (pixels)

    # point - B
    dx_x = abs(point_b[0][0, 0] - point_b[nx - 1][0, 0]) / (nx - 1)
    dy_x = abs(point_b[0][0, 1] - point_b[nx - 1][0, 1]) / (nx - 1)
    xss_b = (dx_x ** 2 + dy_x ** 2) ** 0.5  # x-Size target Square size (pixels)

    dx_y = abs(point_b[0][0, 0] - point_b[(nx * nx) - nx][0, 0]) / (nx - 1)
    dy_y = abs(point_b[0][0, 1] - point_b[(nx * nx) - nx][0, 1]) / (nx - 1)
    yss_b = (dx_y ** 2 + dy_y ** 2) ** 0.5  # y-Size target Square size (pixels)

    # point - C
    dx_x = abs(point_c[0][0, 0] - point_c[nx - 1][0, 0]) / (nx - 1)
    dy_x = abs(point_c[0][0, 1] - point_c[nx - 1][0, 1]) / (nx - 1)
    xss_c = (dx_x ** 2 + dy_x ** 2) ** 0.5  # x-Size target Square size (pixels)

    dx_y = abs(point_c[0][0, 0] - point_c[(nx * nx) - nx][0, 0]) / (nx - 1)
    dy_y = abs(point_c[0][0, 1] - point_c[(nx * nx) - nx][0, 1]) / (nx - 1)
    yss_c = (dx_y ** 2 + dy_y ** 2) ** 0.5  # y-Size target Square size (pixels)

    # point - D
    dx_x = abs(point_d[0][0, 0] - point_d[nx - 1][0, 0]) / (nx - 1)
    dy_x = abs(point_d[0][0, 1] - point_d[nx - 1][0, 1]) / (nx - 1)
    xss_d = (dx_x ** 2 + dy_x ** 2) ** 0.5  # x-Size target Square size (pixels)

    dx_y = abs(point_d[0][0, 0] - point_d[(nx * nx) - nx][0, 0]) / (nx - 1)
    dy_y = abs(point_d[0][0, 1] - point_d[(nx * nx) - nx][0, 1]) / (nx - 1)
    yss_d = (dx_y ** 2 + dy_y ** 2) ** 0.5  # y-Size target Square size (pixels)

    xss = [xss_a, xss_b, xss_c, xss_d]
    yss = [yss_a, yss_b, yss_c, yss_d]

    return xss, yss


def scaleing_conf(resized, nx):
    resized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    # resized = cv2.resize(resized, (0, 0), fx=2, fy=2)

    # resized = cv.multiply(resized, 0.8)
    # resized = cv.multiply(resized, 1.2)

    h, w = resized.shape[:2]

    crop_a = resized[0:int(h / 2), 0:int(w / 2)]
    crop_c = resized[int(h / 2):h, 0:int(w / 2)]
    crop_b = resized[0:int(h / 2), int(w / 2):w]
    crop_d = resized[int(h / 2):h, int(w / 2):w]

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))

    ret_a, point_a = cv.findChessboardCorners(crop_a, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_a:
        ret_a, point_a = cv.findChessboardCorners(clahe.apply(crop_a), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret_b, point_b = cv.findChessboardCorners(crop_b, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_b:
        ret_b, point_b = cv.findChessboardCorners(clahe.apply(crop_b), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret_c, point_c = cv.findChessboardCorners(crop_c, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_c:
        ret_c, point_c = cv.findChessboardCorners(clahe.apply(crop_c), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret_d, point_d = cv.findChessboardCorners(crop_d, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_d:
        ret_d, point_d = cv.findChessboardCorners(clahe.apply(crop_d), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    xss_arr, yss_arr = calc_aspect_ratio(point_a, point_b, point_c, point_d, nx)  # based on one target ONLY!
    xss = sum(xss_arr) / len(xss_arr)
    yss = sum(yss_arr) / len(yss_arr)

    conf_x = s_c_params.ONE_SQUARE_TARGET_SIZE / xss
    conf_y = s_c_params.ONE_SQUARE_TARGET_SIZE / yss

    conf_x = (1.0 - abs(1.0 - conf_x))
    conf_y = (1.0 - abs(1.0 - conf_y))

    return conf_x, conf_y, point_a, point_b, point_c, point_d


def scale_correction(undist, xss, yss, nx):  # pixel to c"m ratio
    resize_y = s_c_params.ONE_SQUARE_TARGET_SIZE / yss / s_c_params.CM_PER_PIXEL
    resize_x = s_c_params.ONE_SQUARE_TARGET_SIZE / xss / s_c_params.CM_PER_PIXEL

    resized = cv.resize(undist, (0, 0), fx=resize_x, fy=resize_y)
    # cv.imwrite('undist_resized.png', resized)

    conf_x, conf_y, point_a, point_b, point_c, point_d = scaleing_conf(resized, nx)

    return resized, conf_x, conf_y, point_a, point_b, point_c, point_d, resize_x, resize_y


def scale(img, nx):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    crop_a = gray[0:int(h / 2), 0:int(w / 2)]
    crop_c = gray[int(h / 2):h, 0:int(w / 2)]
    crop_b = gray[0:int(h / 2), int(w / 2):w]
    crop_d = gray[int(h / 2):h, int(w / 2):w]

    # adaptive histogram
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))

    ret_a, point_a = cv.findChessboardCorners(crop_a, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_a:
        ret_a, point_a = cv.findChessboardCorners(clahe.apply(crop_a), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret_b, point_b = cv.findChessboardCorners(crop_b, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_b:
        ret_b, point_b = cv.findChessboardCorners(clahe.apply(crop_b), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret_c, point_c = cv.findChessboardCorners(crop_c, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_c:
        ret_c, point_c = cv.findChessboardCorners(clahe.apply(crop_c), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    ret_d, point_d = cv.findChessboardCorners(crop_d, (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    if not ret_d:
        ret_d, point_d = cv.findChessboardCorners(clahe.apply(crop_d), (nx, nx), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    # alert if target hasn't detected
    if not ret_a and not ret_b and not ret_c and not ret_d:
        print("All 4-targets hasn't detected")
        return 0.0, 0.0, 0, 0, 0, 0, 1.0, 1.0

    # scale calculation
    xss_arr, yss_arr = calc_aspect_ratio(point_a, point_b, point_c, point_d, nx)  # based on one target ONLY!
    xss = sum(xss_arr) / len(xss_arr)
    yss = sum(yss_arr) / len(yss_arr)

    # alert if final sampling resolution is less than requested
    if xss < 0.9*s_c_params.ONE_SQUARE_TARGET_SIZE or yss < 0.9*s_c_params.ONE_SQUARE_TARGET_SIZE:
        print("FINAL RESOLUTION IS LOWER THAN REQUESTED ! ! !")
        print("Consider redesign camera's position / resolution / FOV.")

    # rescale image & confidence
    img_resized, conf_x, conf_y, point_a, point_b, point_c, point_d, resize_x, resize_y = scale_correction(img, xss, yss, nx)
    print("Scaling conf: ", 100.0*conf_x, " ", 100.0*conf_y)

    return conf_x, conf_y, point_a, point_b, point_c, point_d, resize_x, resize_y


# Write YML file
def save_cam_calibration_params_yaml(filename, image_size, ret, mtx, dist):
    fs_write = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs_write.write("size", image_size)
    fs_write.write("rms", ret)
    fs_write.write("camera_matrix", mtx)
    fs_write.write("dist_coefs", dist)

    fs_write.release()


# Read YML file
def load_cam_calibration_params_yaml(filename):
    fs_read = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    image_size = fs_read.getNode("size")
    ret = fs_read.getNode("rms")
    mtx = fs_read.getNode("camera_matrix")
    dist = fs_read.getNode("dist_coefs")

    # fs_read.release()

    return image_size.mat(), ret.real(), mtx.mat(), dist.mat()


def camera_calibration_proc_video(cam_url, mtx, dist, newcameramtx, resize_x, resize_y, cr_max_x, cr_min_x, cr_max_y, cr_min_y):

    cap = cv.VideoCapture('rtsp://admin:arik12345@192.168.40.243/axis-media/media.amp')
    # cap = cv.VideoCapture(cam_url)
    print('Note: trying open url camera: ', cam_url)

    # cap = cv.VideoCapture(0)

    if cap is None or not cap.isOpened():
        print('Quit running.')
        sys.exit()
    else:
        continue_grabbing = True
        while continue_grabbing:
            ret_cap, frame = cap.read()

            undist = cv.undistort(frame, mtx, dist, None, newcameramtx)
            un_scaled = cv.resize(undist, (0, 0), fx=resize_x, fy=resize_y)

            # cropping
            un_res_crop = un_scaled[int(cr_min_y):int(cr_max_y), int(cr_min_x):int(cr_max_x)]

            frame_res = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            un_res_crop_res = cv.resize(un_res_crop, (0, 0), fx=0.5, fy=0.5)

            cv.imshow("Original", frame_res)
            cv.imshow("undist and scaled", un_res_crop_res)
            cv.waitKey(1)

            if cv.waitKey(10) & 0xFF == ord(' '):
                continue_grabbing = False


def crop_points(undist_crop, point_a, point_b, point_c, point_d, nx, layers_add, resize_x, resize_y):

    undist_crop_scal = cv.resize(undist_crop, (0, 0), fx=resize_x, fy=resize_y)

    h, w = undist_crop_scal.shape[:2]
    p_a = point_a[int((nx*nx) / 2)]
    p_b = point_b[int((nx*nx) / 2)] + [int(w / 2), 0]
    p_c = point_c[int((nx*nx) / 2)] + [0, int(h / 2)]
    p_d = point_d[int((nx*nx) / 2)] + [int(w / 2), int(h / 2)]

    max_x = max([p_b[0, 0], p_d[0, 0]])
    min_x = min([p_a[0, 0], p_c[0, 0]])
    max_y = max([p_c[0, 1], p_d[0, 1]])
    min_y = min([p_a[0, 1], p_b[0, 1]])

    # clamp
    if max_x + s_c_params.LAYERS_GAP < w:
        max_x = max_x + s_c_params.LAYERS_GAP
    else:
        max_x = w
    if min_x - s_c_params.LAYERS_GAP > 0:
        min_x = min_x - s_c_params.LAYERS_GAP
    else:
        min_x = 0
    if max_y + s_c_params.LAYERS_GAP < h:
        max_y = max_y + s_c_params.LAYERS_GAP
    else:
        max_y = h
    if min_y - s_c_params.LAYERS_GAP > 0:
        min_y = min_y - s_c_params.LAYERS_GAP
    else:
        min_y = 0

    # undist_crop_scal_crop = undist_crop_scal[int(min_y):int(max_y), int(min_x):int(max_x)]
    # cv.imshow('undist_crop_scal_crop', undist_crop_scal_crop)
    # cv.waitKey(10)

    return max_x, min_x, max_y, min_y
