import numpy as np
import cv2

def get_undist_params(fn_prefix='./camera_cal/calibration', nx=9, ny=6):
  fnames = []
  objpoints = []
  imgpoints = []
  objp = np.zeros((nx*ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

  for i in range(20):
    fname = fn_prefix + str(i+1) + '.jpg'
    fnames.append(fname)

  for fn in fnames:
    img = cv2.imread(fn)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

    if ret:
      objpoints.append(objp)
      imgpoints.append(corners)

  ret, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                              imgpoints,
                                                              img.shape[:-1][::-1],
                                                              None, None)
  return ret, camMat, distCoeffs, rvecs, tvecs

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
  if orient == 'x':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  elif orient == 'y':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  else:
    raise ValueError('orient must be either x or y')

  abs_sobel = np.absolute(sobel)
  scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

  return binary_output

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  mag = np.sqrt(sobelx**2, sobely**2)
  scaled_sobel = np.uint8(255 * mag / np.max(mag))

  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

  return binary_output

def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  abs_sobelx = np.absolute(sobelx)
  abs_sobely = np.absolute(sobely)

  abs_grad_dir = np.arctan2(abs_sobely, abs_sobelx)

  binary_output = np.zeros_like(abs_grad_dir)
  binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
  return binary_output

def combine_grad(gray, ksize=3, grad_thresh=(20, 100), magnitude_thresh=(30, 100), direction_thresh=(0.7, 1.57)):
  gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
  grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=grad_thresh)
  mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=magnitude_thresh)
  dir_binary = dir_thresh(gray, sobel_kernel=15, thresh=direction_thresh)

  combined = np.zeros_like(dir_binary)
  combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

  return combined

def combine_color(img_bgr, rgb_thresh=(220, 255), hls_thresh=(90, 255)):
  img_r = img_bgr[:, :, 2]
  binary_r = np.zeros_like(img_r)
  binary_r[(img_r > rgb_thresh[0]) & (img_r <= rgb_thresh[1])] = 1

  hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
  S = hls[:, :, 2]
  L = hls[:, :, 1]
  binary_s = np.zeros_like(S)
  binary_l = np.zeros_like(L)
  binary_s[(S > hls_thresh[0]) & (S <= hls_thresh[1])] = 1
  binary_l[(L > hls_thresh[0]) & (L <= hls_thresh[1])] = 1

  combined = np.zeros_like(img_r)
  combined[((binary_s == 1) & (binary_l == 1)) | (binary_r == 1)] = 1

  return combined

def window_mask(width, height, img_ref, center,level):
  output = np.zeros_like(img_ref)
  output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
          max(0,int(center-width/2)):min(int(center+width/2),
                                         img_ref.shape[1])] = 1
  return output

def find_lane_line_pixels(image, window_width, window_height, margin):
  window_centroids = []
  window = np.ones(window_width)
  l_sum = np.sum(image[int(image.shape[0]/2):,:int(image.shape[1]/2)], axis=0)
  l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
  r_sum = np.sum(image[int(image.shape[0]/2):,int(image.shape[1]/2):], axis=0)
  r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
  window_centroids.append((l_center, r_center))
  nonzero = image.nonzero()
  nonzeroy = nonzero[0]
  nonzerox = nonzero[1]

  left_lane_inds = []
  right_lane_inds = []

  win_y_low = image.shape[0] - window_height
  win_y_high = image.shape[0]

  win_x_l_low = l_center - margin
  win_x_l_high = l_center + margin
  win_x_r_low = r_center - margin
  win_x_r_high = r_center + margin

  good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_l_low) & (nonzerox < win_x_l_high)).nonzero()[0]
  good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_r_low) & (nonzerox < win_x_r_high)).nonzero()[0]

  left_lane_inds.append(good_left_inds)
  right_lane_inds.append(good_right_inds)

  for level in range(1,(int)(image.shape[0]/window_height)):
    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
    conv_signal = np.convolve(window, image_layer)
    offset = window_width/2
    l_min_index = int(max(l_center+offset-margin,0))
    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    r_min_index = int(max(r_center+offset-margin,0))
    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    window_centroids.append((l_center,r_center))

    win_y_low = image.shape[0] - (level + 1) * window_height
    win_y_high = image.shape[0] - level * window_height

    win_x_l_low = l_center - margin
    win_x_l_high = l_center + margin
    win_x_r_low = r_center - margin
    win_x_r_high = r_center + margin

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_x_l_low) & (nonzerox < win_x_l_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_x_r_low) & (nonzerox < win_x_r_high)).nonzero()[0]

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)
  ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  return leftx, lefty, rightx, righty
