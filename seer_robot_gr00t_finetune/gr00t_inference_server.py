#!/usr/bin/env python3
"""Standalone GR00T HTTP inference server for Seer live requests."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import threading

from fastapi import FastAPI, HTTPException
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
from PIL import Image
from pydantic import BaseModel
from scipy.spatial.transform import Rotation
import torch
import uvicorn


ACTION_COMPONENTS = {
    "left_ee_pose_target": ("x", "y", "z", "rx", "ry", "rz"),
    "right_ee_pose_target": ("x", "y", "z", "rx", "ry", "rz"),
    "left_gripper_command": (None,),
    "right_gripper_command": (None,),
}


class Request(BaseModel):
    schema: str = "seer.model_inference.request.v1"
    request_id: str
    backend: str = "gr00t"
    task: str = ""
    state: list[float]
    state_names: list[str]
    images: dict[str, str]
    image_encoding: str = "jpeg_base64"
    rtc_options: dict | None = None
    session_id: str | None = None


class Server:
    def __init__(self, model_path: str, device: str, denoising_steps: int):
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=model_path,
            device=device,
        )
        self.policy.model.action_head.num_inference_timesteps = denoising_steps
        self.device = device
        self.action_keys = list(self.policy.modality_configs["action"].modality_keys)
        self.action_horizon = len(self.policy.modality_configs["action"].delta_indices)
        self._active_session: tuple[str | None, str] | None = None
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gr00t-inference")

    @staticmethod
    def _decode(data: str) -> np.ndarray:
        try:
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"invalid image: {exc}") from exc

    @staticmethod
    def _state(req: Request) -> dict[str, np.ndarray]:
        values = dict(zip(req.state_names, req.state, strict=False))

        def get(name: str, default: float = 0.0) -> float:
            if name in values:
                return float(values[name])
            # The collector may include velocity/effort and therefore names
            # joint positions as ``foo.position``. GR00T consumes positions
            # only, so accept both configured spellings.
            return float(values.get(f"{name}.position", default))

        def joints(side: str) -> np.ndarray:
            return np.asarray([get(f"{side}_arm_joint_{i}") for i in range(1, 8)], dtype=np.float32)

        def pose(side: str) -> np.ndarray:
            xyz = [get(f"{side}_ee.{x}") for x in ("x", "y", "z")]
            quat = [get(f"{side}_ee.q{x}") for x in ("x", "y", "z", "w")]
            q = np.asarray(quat, dtype=np.float64)
            if np.linalg.norm(q) < 1e-8:
                q = np.asarray([0, 0, 0, 1.0])
            q /= np.linalg.norm(q)
            return np.asarray(xyz + Rotation.from_quat(q).as_rotvec().tolist(), dtype=np.float32)

        return {
            "left_arm_joint_position": joints("left"),
            "right_arm_joint_position": joints("right"),
            "waist_position": np.asarray([get("waist_position")], dtype=np.float32),
            "lift_position": np.asarray([get("lift_position")], dtype=np.float32),
            "left_gripper_position": np.asarray([get("left_gripper")], dtype=np.float32),
            "right_gripper_position": np.asarray([get("right_gripper")], dtype=np.float32),
            "left_ee_pose": pose("left"),
            "right_ee_pose": pose("right"),
        }

    def infer(self, req: Request) -> dict:
        if req.rtc_options is not None:
            options = req.rtc_options
            if not isinstance(options, dict) or set(options) != {
                "action_horizon",
                "rtc_overlap_steps",
                "rtc_frozen_steps",
                "rtc_ramp_rate",
            }:
                raise ValueError("invalid RTC options")
            horizon = int(options["action_horizon"])
            overlap = int(options["rtc_overlap_steps"])
            frozen = int(options["rtc_frozen_steps"])
            ramp = float(options["rtc_ramp_rate"])
            if (
                horizon != self.action_horizon
                or not 0 < overlap <= horizon
                or not 0 <= frozen <= overlap
                or not np.isfinite(ramp)
                or ramp <= 0
            ):
                raise ValueError("RTC options outside model horizon")
        images = {name: self._decode(data) for name, data in req.images.items()}
        state = self._state(req)
        obs = {
            "video": {
                key: images[key][None, None, ...]
                for key in ("head_image", "left_wrist_image", "right_wrist_image")
            },
            "state": {key: value[None, None, ...] for key, value in state.items()},
            "language": {"annotation.human.action.task_description": [[req.task]]},
        }
        with self.lock, torch.inference_mode():
            session = (req.session_id, req.task)
            if session != self._active_session or req.rtc_options is None:
                self.policy.reset()
            try:
                action, info = self.policy.get_action(obs, options=req.rtc_options)
            except Exception:
                self.policy.reset()
                self._active_session = None
                raise
            self._active_session = session
        arrays = []
        action_names = []
        for key in self.action_keys:
            values = np.asarray(action[key])
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError(
                    f"action {key!r} must have shape (1, horizon, dim); got {values.shape}"
                )
            if values.shape[1] < self.action_horizon:
                raise ValueError(
                    f"action {key!r} horizon {values.shape[1]} is shorter than configured "
                    f"horizon {self.action_horizon}"
                )
            components = ACTION_COMPONENTS.get(key)
            if components is None or len(components) != values.shape[2]:
                raise ValueError(
                    f"unsupported action layout for {key!r}: dimension={values.shape[2]}"
                )
            action_names.extend(
                key if component is None else f"{key}.{component}" for component in components
            )
            arrays.append(values[0, : self.action_horizon])

        action_chunk = np.concatenate(arrays, axis=-1).astype(np.float64).tolist()
        action = action_chunk[0]
        return {
            "schema": "seer.model_inference.response.v1",
            "request_id": req.request_id,
            "backend": "gr00t",
            "status": "ok",
            "action": action,
            "action_chunk": action_chunk,
            "action_horizon": len(action_chunk),
            "action_names": action_names,
            "rtc_supported": True,
            "rtc_rebased": info.get("rtc_rebased", False),
            "rtc_frozen_normalized_max_error": info.get("rtc_frozen_normalized_max_error"),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=4)
    args = ap.parse_args()
    server = Server(args.model_path, args.device, args.denoising_steps)
    app = FastAPI(title="Seer GR00T inference")

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "gr00t"}

    @app.post("/v1/gr00t/action")
    def action(req: Request):
        if req.schema != "seer.model_inference.request.v1":
            raise HTTPException(400, "unsupported request schema")
        try:
            return server.pool.submit(server.infer, req).result()
        except Exception as exc:
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
