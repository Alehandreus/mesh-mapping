import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pygame
import torch
from renderer_utils import initial_camera_from_mesh, camera_rays_from_pose
from main import Renderer
from typing import Dict, Tuple, Any
from main import PyMesh
from main import ResidualMap

# ============================================================
# Interactive progressive rendering loop (pygame)
# ============================================================
@dataclass
class Camera:
    pos: np.ndarray  # (3,)
    yaw: float
    pitch: float

    def forward_right_up(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)

        fwd = np.array([cy * cp, sy * cp, sp], dtype=np.float64)
        fwd /= (np.linalg.norm(fwd) + 1e-12)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(fwd, world_up)
        nr = np.linalg.norm(right)
        if nr < 1e-12:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= nr

        up = np.cross(right, fwd)
        up /= (np.linalg.norm(up) + 1e-12)
        return fwd, right, up


def run_interactive_progressive(
    renderer_obj,
    mesh,
    device: torch.device,
    img_size: int = 512,
    angle0: float = 0.0,
    target_present_fps: int = 60,
    pass_time_budget_ms: float = 10.0,
    mouse_sens: float = 0.0025,   # radians per pixel
    move_speed: float = 2.0,      # units/sec
    sprint_mult: float = 3.0,
):
    """
    Expected renderer API:
        img = renderer_obj.draw(cam_state, cam_delta)

    Where:
      cam_state contains current pose + mesh scale + a rays helper:
        cam_state["rays_fn"](cam_state) -> (d_cam_poses, d_dirs)

      cam_delta tells whether the camera changed since the LAST draw() call.
      Use cam_delta["changed"] to reset accumulation in your renderer when needed.

    img must be (H,W,3) uint8 (RGB).
    """

    init = initial_camera_from_mesh(mesh, angle=angle0)
    center = init["center"]
    max_extent = init["max_extent"]
    dist0 = init["dist0"]

    cam = Camera(pos=init["pos"].copy(), yaw=init["yaw"], pitch=init["pitch"])

    # Track camera state relative to LAST draw() call (important for progressive passes)
    last_call_pos = cam.pos.copy()
    last_call_yaw = cam.yaw
    last_call_pitch = cam.pitch

    pygame.init()
    pygame.display.set_caption("Interactive progressive rendering (pygame)")
    screen = pygame.display.set_mode((img_size, img_size))
    clock = pygame.time.Clock()

    captured = True
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()

    last_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    def rays_fn(cam_state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return camera_rays_from_pose(
            cam_pos=cam_state["pos"],
            yaw=cam_state["yaw"],
            pitch=cam_state["pitch"],
            img_size=cam_state["img_size"],
            max_extent=cam_state["max_extent"],
            dist_ref=cam_state["dist_ref"],
            device=cam_state["device"],
        )

    running = True
    while running:
        dt = clock.tick(target_present_fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                captured = not captured
                pygame.event.set_grab(captured)
                pygame.mouse.set_visible(not captured)
                pygame.mouse.get_rel()

        keys = pygame.key.get_pressed()

        # Mouse look
        if captured:
            dx, dy = pygame.mouse.get_rel()
        else:
            dx, dy = 0, 0

        cam.yaw -= dx * mouse_sens
        cam.pitch -= dy * mouse_sens

        max_pitch = np.deg2rad(89.0)
        cam.pitch = float(np.clip(cam.pitch, -max_pitch, max_pitch))

        # Movement
        fwd, right, up = cam.forward_right_up()
        v = np.zeros(3, dtype=np.float64)

        if keys[pygame.K_w]:
            v += fwd
        if keys[pygame.K_s]:
            v -= fwd
        if keys[pygame.K_d]:
            v += right
        if keys[pygame.K_a]:
            v -= right
        if keys[pygame.K_SPACE]:
            v += up
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            v -= up

        speed = move_speed * (sprint_mult if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0)
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            cam.pos += (v / nv) * speed * dt

        # Delta vs LAST draw() call
        delta_pos = cam.pos - last_call_pos
        delta_yaw = cam.yaw - last_call_yaw
        delta_pitch = cam.pitch - last_call_pitch

        moved = (
            (np.linalg.norm(delta_pos) > 1e-10)
            or (abs(delta_yaw) > 1e-12)
            or (abs(delta_pitch) > 1e-12)
        )

        cam_state = {
            "pos": cam.pos.copy(),
            "yaw": float(cam.yaw),
            "pitch": float(cam.pitch),
            "img_size": int(img_size),
            "center": center,             # available if your renderer wants it
            "max_extent": float(max_extent),
            "dist_ref": float(dist0),     # keep ray scaling stable like the original setup
            "device": device,
            "rays_fn": rays_fn,           # call this to get (d_cam_poses, d_dirs)
        }

        delta_state_first = {
            "pos": delta_pos.copy(),
            "yaw": float(delta_yaw),
            "pitch": float(delta_pitch),
            "changed": bool(moved),
        }
        delta_state_zero = {"pos": np.zeros(3), "yaw": 0.0, "pitch": 0.0, "changed": False}

        # Progressive passes within a time budget
        budget_s = pass_time_budget_ms / 1000.0
        t0 = time.perf_counter()
        passes = 0

        while True:
            if passes == 0:
                cam_delta = delta_state_first
                last_call_pos = cam.pos.copy()
                last_call_yaw = cam.yaw
                last_call_pitch = cam.pitch
            else:
                cam_delta = delta_state_zero

            img = renderer_obj.draw(cam_state, cam_delta)
            passes += 1
            last_img = img

            if (time.perf_counter() - t0) >= budget_s:
                break

        # Present
        if not (isinstance(last_img, np.ndarray) and last_img.ndim == 3 and last_img.shape[2] == 3):
            raise TypeError("renderer_obj.draw must return an (H,W,3) numpy array (uint8 RGB).")

        if last_img.dtype != np.uint8:
            last_img = np.clip(last_img, 0, 255).astype(np.uint8)

        if last_img.shape[0] != img_size or last_img.shape[1] != img_size:
            raise ValueError(f"Expected image shape {(img_size, img_size, 3)}, got {last_img.shape}")

        surf = pygame.surfarray.make_surface(np.swapaxes(last_img, 0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()


# ============================================================
# Example renderer stub showing how to use cam_state["rays_fn"]
# ============================================================
class ExampleRenderer:
    def __init__(self):
        self.acc = None
        self.n = 0

    def draw(self, cam_state: Dict[str, Any], cam_delta: Dict[str, Any]) -> np.ndarray:
        if self.acc is None or cam_delta.get("changed", False):
            H = W = cam_state["img_size"]
            self.acc = np.zeros((H, W, 3), dtype=np.float64)
            self.n = 0

        # Your real renderer would do:
        # d_cam_poses, d_dirs = cam_state["rays_fn"](cam_state)
        # ... trace / shade one pass into an image, accumulate, etc.

        # Toy progressive noise:
        H = W = cam_state["img_size"]
        sample = np.random.rand(H, W, 3)
        self.acc += sample
        self.n += 1
        img = (self.acc / self.n) * 255.0
        return img.astype(np.uint8)


if __name__ == "__main__":
    device = "cuda"

    orig_path = "models/petmonster_orig.fbx"
    inner_path = "models/petmonster_inner_2000.fbx"
    outer_path = "models/petmonster_outer_2000.fbx"

    orig_mesh = PyMesh.from_file(orig_path)
    inner_mesh = PyMesh.from_file(inner_path)
    outer_mesh = PyMesh.from_file(outer_path)

    ckpt_path = "mapping.pt"
    inner_net = ResidualMap(inner_mesh.mesh).to(device)
    outer_net = ResidualMap(outer_mesh.mesh).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    inner_net.load_state_dict(ckpt["inner_net"])
    outer_net.load_state_dict(ckpt["outer_net"])    

    # Usage (replace with your mesh / device / renderer):
    device = torch.device("cuda:0")
    # renderer = ExampleRenderer()
    renderer = Renderer(inner_mesh, outer_mesh, inner_net, outer_net)
    run_interactive_progressive(renderer, inner_mesh.mesh, device, img_size=768)
