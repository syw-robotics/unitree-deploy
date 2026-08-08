# Go2 Proof Loco Heightmap Policy

This checkpoint deploys the teacher policy exported from:

`Vision-Parkour-ZRL-Encoder-TS-EstLinVel-Mask-Exterio-Latent-Teacher-Go2`

The ONNX model consumes 450 proprioceptive values, a 162-point heightmap, and
one exteroception-valid mask. Deployment supplies a live MuJoCo ray-cast
heightmap and fixes the mask to `1.0`.

Start the simulator with the matching sensor configuration:

```bash
unitree-sim-bridge \
  --robot go2 \
  --terrain parkour \
  --sensor ckpt/go2/proof_loco_heightmap/sensor_height_scan.yaml
```

Then start the controller:

```bash
unitree-controller \
  --mode sim \
  --robot go2 \
  --ckpt ckpt/go2/proof_loco_heightmap/policy.yaml
```

The sensor grid and preprocessing intentionally reproduce the training task:

- grid: 18 x 9 points at 0.1 m resolution over 1.7 m x 0.8 m, centered 0.6 m
  ahead of the base;
- observation: `base_z - terrain_z - 0.25`;
- exteroception mask: always valid (`1.0`).
