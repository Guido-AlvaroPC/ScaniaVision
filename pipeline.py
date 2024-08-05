import pickle
import matplotlib.pyplot as plt
import PIL
import pipeline_helpers as ph
import numpy as np
import cv2
import imageio
from moviepy.editor import VideoFileClip

class Line():
  def __init__(self):
    self.detected = False
    self.recent_xfitted = []
    self.bestx = None
    self.best_fit = None
    self.current_fit = [np.array([False])]
    self.radius_of_curvature = None
    self.line_base_pos = None
    self.diffs = np.array([0, 0, 0], dtype='float')
    self.allx = None
    self.ally = None
def get_undist_params():
  try:
    with open('undist_params.p', mode='rb') as f:
      undist_params = pickle.load(f)
      ret, camMat, distCoeffs, rvecs, tvecs = undist_params['ret'], \
                                              undist_params['camMat'], \
                                              undist_params['distCoeffs'], \
                                              undist_params['rvecs'], \
                                              undist_params['tvecs']
  except FileNotFoundError:
    undist_params = {}
    ret, camMat, distCoeffs, rvecs, tvecs = ph.get_undist_params()
    undist_params['ret'], undist_params['camMat'], undist_params['distCoeffs'], \
        undist_params['rvecs'], undist_params['tvecs'] = ret, camMat, distCoeffs, rvecs, tvecs

    with open('undist_params.p', mode='wb') as f:
      pickle.dump(undist_params, f)

  return ret, camMat, distCoeffs, rvecs, tvecs

def get_all_imgs(img, is_bgr):
  if is_bgr:
    img_bgr = img
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  else:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  return img, img_bgr, img_gray


def get_perspective_mat(camMat, distCoeffs, img_size):
  try:
    with open('perspective_params.p', mode='rb') as f:
      perspective_params = pickle.load(f)
      matP, matP_inv = perspective_params['matP'], perspective_params['matP_inv']
  except FileNotFoundError:
    straight_line_bgr = cv2.imread('./test_images/straight_lines2.jpg')
    undist_straight_line = cv2.undistort(straight_line_bgr, camMat, distCoeffs, None, camMat)

    upper_left = [578, 462]
    upper_right = [704, 462]
    lower_left = [212, 719]
    lower_right = [1093, 719]

    offset_x = 300
    offset_y = 0

    src = np.float32([upper_left, upper_right, lower_right, lower_left])
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x - 1, offset_y],
                      [img_size[0] - offset_x - 1, img_size[1] - offset_y - 1],
                      [offset_x, img_size[1] - offset_y - 1]])

    matP = cv2.getPerspectiveTransform(src, dst)
    matP_inv = cv2.getPerspectiveTransform(dst, src)

    perspective_params = {}
    perspective_params['matP'], perspective_params['matP_inv'] = matP, matP_inv
    with open('perspective_params.p', mode='wb') as f:
      pickle.dump(perspective_params, f)

  return matP, matP_inv


def get_bin_lane_line_img(img_gray, img_bgr):
  combined_color = ph.combine_color(img_bgr)
  return combined_color

def color_warped_lane_lines(warped, leftx, lefty, rightx, righty):
  warped[lefty, leftx] = [255, 0, 0]
  warped[righty, rightx] = [0, 0, 255]
  return warped


def get_lane_line_bounded_image(warped, left_fitx, right_fitx, ploty, margin):
  window_img = np.zeros_like(warped)
  left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
  left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                ploty])))])
  left_line_pts = np.hstack((left_line_window1, left_line_window2))
  right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
  right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                ploty])))])
  right_line_pts = np.hstack((right_line_window1, right_line_window2))
  cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
  cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
  return window_img

def get_cuvature(leftx, lefty, rightx, righty, ploty):
  ym_per_pix = 30 / 720
  xm_per_pix = 3.7 / 700
  y_eval = np.max(ploty)

  left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
  left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
  right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

  return left_curverad, right_curverad


def get_car_offset(combined, matP, img_size, bottom_left_fitx, bottom_right_fitx, xm_per_pix):
  car_center = np.zeros_like(combined)
  car_center[car_center.shape[0] - 1, car_center.shape[1] // 2 - 1] = 1
  car_center_warp = cv2.warpPerspective(car_center, matP, (img_size[0], img_size[1]))
  car_centerx = np.argmax(car_center_warp[car_center_warp.shape[0] - 1, :])
  lane_centerx = ((bottom_right_fitx + bottom_left_fitx) // 2)
  car_offset_meters = (car_centerx - lane_centerx) * xm_per_pix
  return car_offset_meters


def color_unwarped_lane(warped, img_size, left_fitx, right_fitx, ploty, matP_inv):
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))
  cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
  newwarp = cv2.warpPerspective(color_warp, matP_inv, (img_size[0], img_size[1]))
  return newwarp

def paste_curvature_and_offset(image, curverad, offset):
  font = cv2.FONT_HERSHEY_SIMPLEX
  image = cv2.putText(image, "Linha de Curvatura: " + str(curverad) + " metros", (20, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
  image = cv2.putText(image, "Distancia: " + str(offset) + " Metros", (20, 120), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
  return image

def update_line(line, fitx, fit):
  line.detected = True
  num_tracked_lines = len(line.recent_xfitted)
  if num_tracked_lines == 10:
    line.recent_xfitted.pop(0)
  line.recent_xfitted.append(fitx)
  line.bestx = np.mean(line.recent_xfitted, axis=0)
  if line.best_fit is None:
    line.best_fit = fit
  else:
    if num_tracked_lines == 10:
      line.best_fit = (line.best_fit * num_tracked_lines + fit) / num_tracked_lines
    else:
      line.best_fit = (line.best_fit * num_tracked_lines + fit) / (num_tracked_lines + 1)

  line.diffs = fit - line.current_fit
  line.current_fit = fit
def process_frames(is_bgr=True, left_line=None, right_line=None):
  if left_line is None:
    left_line = Line()
  if right_line is None:
    right_line = Line()

  def process_image(img):
    img, img_bgr, img_gray = get_all_imgs(img, is_bgr)
    img_size = (img.shape[1], img.shape[0])
    ret, camMat, distCoeffs, rvecs, tvecs = get_undist_params()
    matP, matP_inv = get_perspective_mat(camMat, distCoeffs, img_size)
    undist = cv2.undistort(img, camMat, distCoeffs, None, camMat)
    combined = get_bin_lane_line_img(img_gray, img_bgr)
    warped = cv2.warpPerspective(combined, matP, (img_size[0], img_size[1]))
    window_width = 50
    window_height = 180
    margin = 100
    xm_per_pix = 3.7 / 700
    leftx, lefty, rightx, righty = ph.find_lane_line_pixels(warped, window_width, window_height, margin)
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_curverad, right_curverad = get_cuvature(leftx, lefty, rightx, righty, ploty)
    curverad = (left_curverad + right_curverad) / 2
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    update_line(left_line, left_fitx, left_fit)
    update_line(right_line, right_fitx, right_fit)
    car_offset = get_car_offset(combined, matP, img_size, left_fitx[-1], right_fitx[-1], xm_per_pix)
    colored_lane = color_unwarped_lane(warped, img_size, left_line.bestx, right_line.bestx, ploty, matP_inv)
    colored_lane_img = cv2.addWeighted(undist, 1, colored_lane, 0.3, 0)
    colored_lane_img = paste_curvature_and_offset(colored_lane_img, curverad, car_offset)
    return colored_lane_img
  return process_image

if __name__ == '__main__':
  clip1 = VideoFileClip("./video_teste.mp4")
  left_line = Line()
  right_line = Line()
  result = clip1.fl_image(process_frames(is_bgr=False, left_line=left_line, right_line=right_line))
  result.write_videofile('./video_renderizado.mp4', audio=False)
