"""Live observation -> model HTTP request -> safe action topic service."""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any
import uuid

from data_collection.settings import load_config
from data_collection.topics import RbkTopicSource
import numpy as np
from PIL import Image
import requests

from .config import ModelRequestConfig


class ModelRequestService:
    """Run model requests without touching the robot control topic.

    The returned action is published as JSON on ``response_topic`` only. A
    separate, explicitly reviewed controller bridge must consume it before any
    robot command can be sent.
    """

    def __init__(self, config: ModelRequestConfig, *, fake_data: bool = False):
        self.config = config
        self.collection = load_config(config.collection_config)
        self.fake_data = fake_data
        self.source = None if fake_data else RbkTopicSource(self.collection)
        self._message_type: type | None = None
        self._datapool: Any = None
        self._last_response: dict[str, Any] | None = None

    def start(self) -> None:
        if self.fake_data:
            return
        assert self.source is not None
        self.source.start()
        self._datapool = self.source._datapool
        from robokit.protocol.messageV4_universal_pb2 import MessageV4_UniversalDatas

        self._message_type = MessageV4_UniversalDatas
        self._datapool.publish(self.config.request_topic, self._message_type)
        self._datapool.publish(self.config.response_topic, self._message_type)

    def close(self) -> None:
        if self.source is not None:
            self.source.close()

    @staticmethod
    def _image_b64(image: Any, quality: int) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _request_payload(self, sample: Any) -> dict[str, Any]:
        return {
            "schema": "seer.model_inference.request.v1",
            "request_id": str(uuid.uuid4()),
            "backend": self.config.backend,
            "task": self.config.task,
            "captured_unix_ns": sample.captured_unix_ns,
            "state": sample.state_vector(
                self.collection.include_joint_velocity,
                self.collection.include_joint_effort,
                self.collection.include_ee_pose,
                self.collection.gripper.state_as_observation,
            ).tolist(),
            "state_names": list(self.collection.state_names),
            "images": {
                name: self._image_b64(image, self.config.image_quality)
                for name, image in sample.images.items()
            },
            "image_encoding": "jpeg_base64",
            "action_names": list(self.collection.action_names),
        }

    def _fake_payload(self) -> dict[str, Any]:
        """Build a safe static observation without initializing Robokit."""

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 64
        image[:, :, 1] = 96
        image[:, :, 2] = 128
        state_names = list(self.collection.state_names)
        state = np.zeros(len(state_names), dtype=np.float32)
        values = dict(zip(state_names, state, strict=True))
        # Identity quaternion for both EEF poses; preserve the configured
        # state vector layout while making the fake request deterministic.
        for side in ("left", "right"):
            values[f"{side}_ee.qw"] = 1.0
        state = np.asarray([values[name] for name in state_names], dtype=np.float32)
        return {
            "schema": "seer.model_inference.request.v1",
            "request_id": str(uuid.uuid4()),
            "backend": self.config.backend,
            "task": self.config.task,
            "captured_unix_ns": time.time_ns(),
            "state": state.tolist(),
            "state_names": state_names,
            "images": {
                name: self._image_b64(image, self.config.image_quality)
                for name in ("head_image", "left_wrist_image", "right_wrist_image")
            },
            "image_encoding": "jpeg_base64",
            "action_names": list(self.collection.action_names),
            "fake_data": True,
        }

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self._datapool is None or self._message_type is None:
            return
        message = self._message_type()
        item = message.data.add()
        item.key = "model_inference"
        item.type = "json"
        item.string_value = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self._datapool.publish(topic, self._message_type)
        # The robokit datapool uses put() for an already-created message in
        # some releases and publish(topic, message) in others. Support both.
        try:
            self._datapool.put(topic, message)
        except (AttributeError, TypeError):
            self._datapool.publish(topic, message)

    def step(self) -> dict[str, Any] | None:
        if self.fake_data:
            payload = self._fake_payload()
        else:
            assert self.source is not None
            sample = self.source.snapshot()
            if sample is None:
                return None
            payload = self._request_payload(sample)
        self._publish(self.config.request_topic, payload)
        backend = self.config.backends[self.config.backend]
        if self.config.dry_run or not backend.url:
            response = {
                "schema": "seer.model_inference.response.v1",
                "request_id": payload["request_id"],
                "backend": self.config.backend,
                "status": "dry_run",
                "action": None,
            }
        else:
            started = time.monotonic_ns()
            try:
                result = requests.post(
                    backend.url,
                    json=payload,
                    timeout=backend.timeout_ms / 1000.0,
                )
                result.raise_for_status()
                body = result.json()
                response = {
                    "schema": "seer.model_inference.response.v1",
                    "request_id": payload["request_id"],
                    "backend": self.config.backend,
                    "status": "ok",
                    "latency_ms": (time.monotonic_ns() - started) / 1e6,
                    "action": body.get("action"),
                    "action_names": body.get("action_names", self.collection.action_names),
                }
            except Exception as exc:
                response = {
                    "schema": "seer.model_inference.response.v1",
                    "request_id": payload["request_id"],
                    "backend": self.config.backend,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": None,
                }
        self._last_response = response
        if self.config.publish_action:
            self._publish(self.config.response_topic, response)
        return response

    def run(self, stop_event: Any) -> None:
        period = 1.0 / self.config.request_hz
        while not stop_event.is_set():
            started = time.monotonic()
            response = self.step()
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                stop_event.wait(remaining)
