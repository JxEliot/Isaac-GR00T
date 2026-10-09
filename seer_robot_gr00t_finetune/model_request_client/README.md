# Model Request Service

This is an independent observation-to-model request process for real-machine
validation. It reuses the data-collection topic subscriptions, encodes the
three camera images as JPEG, and sends a JSON request to either a `pi05` or
`gr00t` model server.

The default mode is deliberately safe:

- `dry_run=true`: no HTTP model call is made;
- `publish_action=false`: no predicted action is published;
- the service never publishes `/RobotControl/WbcEECommand`.

Configure the model server URLs in `config/model_request.json`, then run:

```bash
MODEL_BACKEND=pi05 ./run_model_request.sh
MODEL_BACKEND=gr00t ./run_model_request.sh
```

For a local protocol-only test, without Robokit or a controller:

```bash
MODEL_BACKEND=gr00t MODEL_FAKE_DATA=1 MODEL_LIVE=1 \
  MODEL_URL=http://127.0.0.1:18081/v1/gr00t/action \
  ./run_model_request.sh
```

`MODEL_FAKE_DATA=1` generates deterministic state and camera images. It does
not subscribe to any controller topic.

To enable a real HTTP request and publish the response to the isolated model
topic, use `MODEL_LIVE=1 MODEL_PUBLISH_ACTION=1`. The response topic is
`/ModelInference/Action`; a separate safety-reviewed bridge is required before
any robot command can be sent.

The HTTP request schema is `seer.model_inference.request.v1`. The server must
return JSON containing `action`, and may return `action_names`. PI05 should
return 14 values in its absolute EEF layout:

```text
left xyz+rotvec (6), right xyz+rotvec (6), left gripper, right gripper
```

GR00T should return the action layout declared by the active data-collection
configuration. Keep model inference outside the Quest/teleoperation process so
inference latency cannot block tracking or recording.
