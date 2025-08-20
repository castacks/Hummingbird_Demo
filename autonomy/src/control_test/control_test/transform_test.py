
import numpy as np
from wire_control_test import WireTransforms
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.transform import Rotation as R

pio.renderers.default = "browser"

def test_wire_transform_1():
    wire_transforms = WireTransforms(1.0, 2.0, 3.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    drone_pos = np.array([0.0, 0.0, 0.0])
    drone_quat = np.array([0.0, 0.0, 0.0, 1.0])
    wire_pos_body, wire_yaw_body = wire_transforms.transform_wire_local_to_body(drone_pos, drone_quat)
    assert np.allclose(wire_pos_body, np.array([1.0, 2.0, 3.0]))
    assert np.isclose(wire_yaw_body, 0.0)
    print("test_wire_transform_1 passed")
    print(f"Wire body position: {wire_pos_body}")
    print(f"Wire body yaw: {wire_yaw_body}")

def test_wire_transform_2():
    wire_transforms = WireTransforms(1.0, 1.0, 1.0, 0.0, [0.0, 0.0, 0.0], [ 0, 0, 0.7071068, 0.7071068]) # 90 degree rotation
    drone_pos = np.array([0.0, 0.0, 0.0]) 
    # drone_quat = np.array([ 0, 0, 0.6427863, 0.7660455 ]) # 80 degree rotation
    # drone_quat = np.array([ 0, 0, 0.7660455, 0.6427863 ]) # 100 degree rotation
    drone_quat = np.array([ 0, 0, 0, 1 ]) # 90 degree rotation

    wire_pos_body, wire_yaw_body = wire_transforms.transform_wire_local_to_body(drone_pos, drone_quat)
    print(f"Wire body position: {wire_pos_body}")
    print(f"Wire body yaw: {np.rad2deg(wire_yaw_body)} degrees")
    visualize_wire_transform(drone_pos, drone_quat, wire_transforms.wire_pos, wire_pos_body, wire_yaw_body)

def test_wire_transform_3():
    wire_transforms = WireTransforms(1.0, 1.0, 1.0, 0.0, [0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, 1.0])
    drone_pos = np.array([0.0, 0.0, 0.0]) 
    drone_quat = np.array([ 0, 0, 0.0871557, 0.9961947 ]) # 10 degree rotation
    # drone_quat = np.array([ 0, 0, 0.7071068, 0.7071068 ]) # 90 degree rotation
    wire_pos_body, wire_yaw_body = wire_transforms.transform_wire_local_to_body(drone_pos, drone_quat)
    print(f"Wire body position: {wire_pos_body}")
    print(f"Wire body yaw: {np.rad2deg(wire_yaw_body)} degrees")
    visualize_wire_transform(drone_pos, drone_quat, wire_transforms.wire_pos, wire_pos_body, wire_yaw_body)

def visualize_wire_transform(drone_pos, drone_quat, wire_pos_world, wire_pos_body, wire_yaw_body):
    fig = go.Figure()

    # --- Plot drone ---
    fig.add_trace(go.Scatter3d(
        x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]],
        mode="markers+text",
        marker=dict(size=6, color="blue"),
        text=["Drone"],
        textposition="top center",
        name="Drone"
    ))

    # --- Plot wire in world frame ---
    fig.add_trace(go.Scatter3d(
        x=[wire_pos_world[0]], y=[wire_pos_world[1]], z=[wire_pos_world[2]],
        mode="markers+text",
        marker=dict(size=6, color="red"),
        text=["Wire (world)"],
        textposition="top center",
        name="Wire (world)"
    ))

    # --- Plot wire in drone body frame ---
    fig.add_trace(go.Scatter3d(
        x=[wire_pos_body[0]], y=[wire_pos_body[1]], z=[wire_pos_body[2]],
        mode="markers+text",
        marker=dict(size=6, color="green"),
        text=[f"Wire (body)\nYaw={np.rad2deg(wire_yaw_body):.1f}°"],
        textposition="top center",
        name="Wire (body)"
    ))

    # --- Add drone orientation arrow ---
    r_drone = R.from_quat(drone_quat)
    forward_vec = r_drone.apply([1, 0, 0])  # drone x-axis
    fig.add_trace(go.Cone(
        x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]],
        u=[forward_vec[0]], v=[forward_vec[1]], w=[forward_vec[2]],
        sizemode="absolute", sizeref=0.5,
        anchor="tail", colorscale="Blues", showscale=False,
        name="Drone Forward"
    ))

    # --- Add wire yaw orientation (in body frame) ---
    wire_dir = [np.cos(wire_yaw_body), np.sin(wire_yaw_body), 0]
    fig.add_trace(go.Cone(
        x=[wire_pos_body[0]], y=[wire_pos_body[1]], z=[wire_pos_body[2]],
        u=[wire_dir[0]], v=[wire_dir[1]], w=[wire_dir[2]],
        sizemode="absolute", sizeref=0.5,
        anchor="tail", colorscale="Reds", showscale=False,
        name="Wire Orientation"
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        title="Drone and Wire Transform Visualization"
    )
    fig.show()


if __name__ == "__main__":
    test_wire_transform_2()
    # test_wire_transform_3()