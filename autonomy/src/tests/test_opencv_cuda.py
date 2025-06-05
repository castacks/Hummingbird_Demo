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
        
if __name__ == "__main__":
    test_opencv_cuda()