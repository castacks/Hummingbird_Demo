import cv2
import numpy as np
import time 

def load_image(img_path):
    # You can replace this with a path to your own image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def run_cpu_hough(image, iterations=50):
    times = []
    for _ in range(iterations):
        start = time.time()
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=10)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

def run_cuda_hough(image, iterations=50):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA device not found or OpenCV not built with CUDA")
        return None

    # Canny on GPU
    times = []
    gpu_image = cv2.cuda_GpuMat()
    gpu_canny = cv2.cuda.createCannyEdgeDetector(50, 150)
    hough_detector = cv2.cuda.createHoughSegmentDetector(1, np.pi / 180, 80, 30, 10)

    for _ in range(iterations):
        start = time.time()
        gpu_image.upload(image)
        gpu_edges = gpu_canny.detect(gpu_image)
        gpu_lines = hough_detector.detect(gpu_edges)
        gpu_lines.download()
        end = time.time()
        times.append(end - start)
    return np.mean(times)
        
if __name__ == "__main__":
    image_path = "./test_img.png"
    iterations = 100
    image = load_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    cpu_time = run_cpu_hough(image, iterations)
    print(f"CPU average iteration time: {cpu_time:.6f} seconds")
    
    gpu_time = run_cuda_hough(image, iterations)
    print(f"GPU average iteration time: {gpu_time:.6f} seconds" if gpu_time is not None else "GPU iterations time: CUDA not available or failed")

    if gpu_time > cpu_time:
        print(f"GPU is slower than CPU by {gpu_time / cpu_time:.4f} times")
    else:
        print(f"GPU is faster than CPU by {cpu_time / gpu_time:.4f} times")
    
