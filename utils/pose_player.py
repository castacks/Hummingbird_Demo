import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from scipy.spatial.transform import Rotation as R

pio.renderers.default = "browser"

# -------- Parameters --------
filename = "/home/tyler/Documents/field_tests/phantom_wire_tests/wire_tracking_20250822_161536/pose_data.csv"  # CSV file with: timestamp,x,y,z,qx,qy,qz,qw

# -------- Load Data --------
data = pd.read_csv(filename)
timestamps = data["timestamp"].values
positions = data[["x", "y", "z"]].values
quaternions = data[["qx", "qy", "qz", "qw"]].values

# -------- Create Frames --------
frames = []
colors = ['red', 'green', 'blue']

for i in range(len(timestamps)):
    pos = positions[i]
    quat = quaternions[i]

    # Orientation
    rot = R.from_quat(quat).as_matrix()
    length = 0.2
    axes = rot @ (np.eye(3) * length) + pos.reshape(3,1)

    frame_data = [
        go.Scatter3d(
            x=positions[:i+1,0], y=positions[:i+1,1], z=positions[:i+1,2],
            mode="lines",
            line=dict(color="black", width=4),
            name="trajectory"
        ),
        go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="current"
        )
    ]

    # Add orientation axes
    for j, c in enumerate(colors):
        frame_data.append(
            go.Scatter3d(
                x=[pos[0], axes[0,j]],
                y=[pos[1], axes[1,j]],
                z=[pos[2], axes[2,j]],
                mode="lines",
                line=dict(color=c, width=6),
                name=f"axis {j}"
            )
        )

    frames.append(go.Frame(data=frame_data, name=str(i)))

# -------- Base Figure --------
fig = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), 
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                        mode="immediate")])
            ]
        )]
    ),
    frames=frames
)

# Add a slider
sliders = [dict(
    steps=[dict(method="animate",
                args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=str(round(timestamps[k],2)))
           for k in range(len(timestamps))],
    active=0
)]
fig.update_layout(sliders=sliders)

fig.show()
