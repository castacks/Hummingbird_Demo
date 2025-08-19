import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# -------- Parameters --------
filename = "/media/tyler/Storage/field_tests/250815_vtolwire_2/pose_data.csv"  
# CSV file with: timestamp,x,y,z,qx,qy,qz,qw

# -------- Load Data --------
data = pd.read_csv(filename)
positions = data[["x", "y", "z"]].values
timestamps = data["timestamp"].values

# Normalize timestamps to [0,1] for colormap
t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

# -------- Create Figure --------
fig = go.Figure()

# Trajectory line with gradient color
fig.add_trace(go.Scatter3d(
    x=positions[:,0],
    y=positions[:,1],
    z=positions[:,2],
    mode="lines",
    line=dict(
        color=t_norm,          # use normalized time for coloring
        colorscale="Agsunset",    # from dark red (start) to light red (end)
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

# Layout
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    title="3D Trajectory with Time Gradient"
)

fig.show()
