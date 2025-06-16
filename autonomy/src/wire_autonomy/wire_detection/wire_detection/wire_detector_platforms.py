import numpy as np
import cv2
from scipy.stats import circmean
from scipy.signal import find_peaks

from wire_detector_base import WireDetector

class WireDetectorCPU(WireDetector):
    def __init__(self, wire_detection_config, camera_intrinsics):
        super().__init__(wire_detection_config, camera_intrinsics)

    def get_hough_lines(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        seg_mask = cv2.Canny(gray, self.low_canny_threshold, self.high_canny_threshold, apertureSize=3)
        cartesian_lines = cv2.HoughLinesP(seg_mask, 1, np.pi/180, self.hough_vote_threshold, minLineLength=self.min_line_threshold, maxLineGap=10)
        if cartesian_lines is None:
            return None
        
        cartesian_lines = np.squeeze(cartesian_lines, axis=1)
        return cartesian_lines
    
    def get_xy_depth_gradients(self, depth_image):
        """
        Compute the x and y gradients of the depth image using Sobel filters.
        """
        depth_gradient_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=11)
        depth_gradient_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=11)
        return depth_gradient_x, depth_gradient_y

class WireDetectorGPU(WireDetector):
    def __init__(self, wire_detection_config, camera_intrinsics):
        super().__init__(wire_detection_config, camera_intrinsics)

        # Persistent GPU Mats
        self.gpu_rgb = cv2.cuda_GpuMat()
        self.gpu_depth = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()

        # Persistent CUDA filters
        self.canny_detector = cv2.cuda.createCannyEdgeDetector(
            self.low_canny_threshold, self.high_canny_threshold, apperture_size=3
        )
        self.hough_detector = cv2.cuda.createHoughSegmentDetector(
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_vote_threshold,
            minLineLength=self.min_line_threshold,
            maxLineGap=10
        )

        self.sobel_x = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 1, 0, ksize=11)
        self.sobel_y = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 0, 1, ksize=11)

    def get_hough_lines(self, rgb_image):
        self.gpu_rgb.upload(rgb_image)
        self.gpu_gray.upload(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY))
        gpu_edges = self.canny_detector.detect(self.gpu_gray)
        gpu_lines = self.hough_detector.detect(gpu_edges)

        if gpu_lines is None or gpu_lines.empty():
            return None
        else:
            cartesian_lines = gpu_lines.download()
            cartesian_lines = np.squeeze(cartesian_lines, axis=0)
            return cartesian_lines
    
    def get_xy_depth_gradients(self, depth_image):
        self.gpu_depth.upload(depth_image)
        depth_gradient_x = self.sobel_x.apply(self.gpu_depth)
        depth_gradient_y = self.sobel_y.apply(self.gpu_depth)
        depth_gradient_x = depth_gradient_x.download()
        depth_gradient_y = depth_gradient_y.download()
        return depth_gradient_x, depth_gradient_y