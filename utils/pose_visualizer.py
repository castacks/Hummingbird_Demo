import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.transform import Rotation as R

pio.renderers.default = "browser"

# -------- Parameters --------
filename = "/media/tyler/Storage/field_tests/250815_vtolwire_1/pose_data.csv"  
# CSV file with: timestamp,x,y,z,qx,qy,qz,qw

# -------- Load Data --------
data = pd.read_csv(filename)
positions = data[["x", "y", "z"]].values
timestamps = data["timestamp"].values
quaternions = data[["qx","qy","qz","qw"]].values

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
