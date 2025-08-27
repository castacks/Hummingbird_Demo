# !/bin/bash
set -e

gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/wire_tracking latency=0 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
