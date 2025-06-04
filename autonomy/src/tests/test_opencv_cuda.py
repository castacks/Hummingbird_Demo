import cv2
import numpy as np

def test_opencv_cuda():
    # Check if OpenCV was built with CUDA
    print(f"OpenCV version: {cv2.__version__}")
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA is not available in OpenCV.")
        return

    print(f"CUDA-enabled device count: {cv2.cuda.getCudaEnabledDeviceCount()}")
    
    # Try to use the first CUDA device
    try:
        cv2.cuda.setDevice(0)
        device_info = cv2.cuda.DeviceInfo(0)
        print(f"Using CUDA device: {device_info.name()}")
    except Exception as e:
        print(f"Failed to set CUDA device: {e}")
        return

    # Create a test image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Upload to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    # Perform a Gaussian Blur on the GPU
    try:
        gpu_blurred = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.5).apply(gpu_img)
        # Download result back to CPU
        result = gpu_blurred.download()
        print("CUDA Gaussian blur succeeded.")
    except Exception as e:
        print(f"CUDA operation failed: {e}")
        return

    # Optionally show the result
    # cv2.imshow('Blurred Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    test_opencv_cuda()
