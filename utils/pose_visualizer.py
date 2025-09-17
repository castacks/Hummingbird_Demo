import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.transform import Rotation as R

pio.renderers.default = "browser"

# -------- Parameters --------
files = ["/home/tyler/Documents/MSR/field_tests/250828_afca/split_data/wire_data_20250828_092730/pose_data.csv"]

# CSV file with: timestamp,x,y,z,qx,qy,qz,qw
final_output = []
for filename in files: 
    # -------- Load Data --------
    data = pd.read_csv(filename)
    positions = data[["x", "y", "z"]].values[1:]  # skip first row
    timestamps = data["timestamp"].values[1:]  # skip first row
    quaternions = data[["qx","qy","qz","qw"]].values[1:]  # skip first row
    yaws = R.from_quat(quaternions).as_euler('zyx')[:,0]

    origin_timestamp = data["timestamp"].values[0]  # first row timestamp
    origin_position = data[["x","y","z"]].values[0]  # first row position
    origin_quaternion = data[["qx","qy","qz","qw"]].values[0]  # first row quaternion
    origin_yaw = R.from_quat(origin_quaternion).as_euler('zyx')[0]

    end_timestamp = data["timestamp"].values[-1]  # last row timestamp
    end_position = data[["x","y","z"]].values[-1]  # last row position
    end_quaternion = data[["qx","qy","qz","qw"]].values[-1]  # last row quaternion
    end_yaw = R.from_quat(end_quaternion).as_euler('zyx')[0]

    final_output.append({
        "filename": filename.split('/')[-2].split('_')[-1],
        "origin": {
            "timestamp": origin_timestamp,
            "position": origin_position,
            "yaw": origin_yaw
        },
        "end": {
            "timestamp": end_timestamp,
            "position": end_position,
            "yaw": end_yaw
        },
        "difference": {
            "time": (end_timestamp - origin_timestamp) / 1e9,  # convert ns to s
            "position": end_position - origin_position,
            "yaw": end_yaw - origin_yaw
        }
    })

    # Normalize timestamps to [0,1] for colormap
    t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

    # -------- Compute orientation vectors --------
    rotations = R.from_quat(quaternions)

    # For visualization, take the vehicle's forward vector (e.g., x-axis in body frame)
    # and rotate it into world frame
    forward_vectors = rotations.apply(np.array([[1, 0, 0]]*len(quaternions)))

    # Subsample vectors for clarity (optional)
    step = max(len(positions)//150, 1)  # show ~150 cones
    positions_sub = positions[::step]
    vectors_sub = forward_vectors[::step]

    # -------- Create Figure --------
    fig = go.Figure()

    # Trajectory line with gradient color
    fig.add_trace(go.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode="lines",
        line=dict(
            color=t_norm,
            colorscale="Agsunset",
            width=4
        ),
        name="trajectory"
    ))

    # Start point
    fig.add_trace(go.Scatter3d(
        x=[positions[0,0]], y=[positions[0,1]], z=[positions[0,2]],
        mode="markers",
        marker=dict(size=6, color="green"),
        name="start"
    ))

    # End point
    fig.add_trace(go.Scatter3d(
        x=[positions[-1,0]], y=[positions[-1,1]], z=[positions[-1,2]],
        mode="markers",
        marker=dict(size=6, color="red"),
        name="end"
    ))

    # Orientation cones
    fig.add_trace(go.Cone(
        x=positions_sub[:,0],
        y=positions_sub[:,1],
        z=positions_sub[:,2],
        u=vectors_sub[:,0],
        v=vectors_sub[:,1],
        w=vectors_sub[:,2],
        sizemode="absolute",
        sizeref=1.0,  # adjust size
        anchor="tail",
        colorscale="Agsunset",
        showscale=False,
        name="orientation"
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        title="3D Trajectory with Orientation"
    )

    fig.show()

rows = []
for entry in final_output:
    rows.append({
        "filename": entry["filename"],
        "origin_timestamp": entry["origin"]["timestamp"],
        "origin_x": entry["origin"]["position"][0],
        "origin_y": entry["origin"]["position"][1],
        "origin_z": entry["origin"]["position"][2],
        "origin_yaw": entry["origin"]["yaw"],
        "end_timestamp": entry["end"]["timestamp"],
        "end_x": entry["end"]["position"][0],
        "end_y": entry["end"]["position"][1],
        "end_z": entry["end"]["position"][2],
        "end_yaw": entry["end"]["yaw"],
        "diff_time": entry["difference"]["time"],
        "diff_x": entry["difference"]["position"][0],
        "diff_y": entry["difference"]["position"][1],
        "diff_z": entry["difference"]["position"][2],
        "diff_yaw": entry["difference"]["yaw"],
    })

df = pd.DataFrame(rows)
df.to_csv("final_analysis.csv", index=False)
