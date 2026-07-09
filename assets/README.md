# Asset Usage Notes

This folder contains the USD and YAML assets used by the Isaac Sim scripts.

## BlueROVHeavy Load Chain

Current code loads the BlueROV2 Heavy model through the small wrapper USD:

```text
EDMDc/*.py
  -> assets/BlueROVHeavy/BlueROVHeavy.usd
      -> ../../bluerov_heavy_cad/bluerov_heavy.usd
          -> assets/BlueROVHeavy/bluerov_heavy_fine.usd
```

The Python scripts build the vehicle asset path from the vehicle name:

```python
VEHICLE = "BlueROVHeavy"
ASSET_DIR = PROJECT / "assets" / VEHICLE
VEHICLE_USD = ASSET_DIR / f"{VEHICLE}.usd"
VEHICLE_YAML = ASSET_DIR / f"{VEHICLE}.yaml"
```

Then Isaac Sim references `VEHICLE_USD` into the stage, usually at
`/World/Vehicle`.

## What Each File Does

- `assets/BlueROVHeavy/BlueROVHeavy.usd`
  - Small Isaac-ready wrapper USD.
  - Provides the expected vehicle structure, including `base_link`, physics
    setup, collision/visual organization, and the reference to the CAD mesh.
  - This is the USD file that current Python code loads directly.

- `assets/BlueROVHeavy/BlueROVHeavy.yaml`
  - Hydrodynamic and thruster configuration used by the Fossen model,
    allocator, and controller code.

- `assets/BlueROVHeavy/bluerov_heavy_fine.usd`
  - Large high-detail CAD/visual USD.
  - This is the visually detailed mesh, but current control code should not
    load it directly because it may not provide the expected Isaac vehicle
    prim structure.

- `bluerov_heavy_cad/bluerov_heavy.usd`
  - Compatibility path referenced by `BlueROVHeavy.usd`.
  - In this workspace it is a symlink to
    `assets/BlueROVHeavy/bluerov_heavy_fine.usd`.

## Upload Recommendation

For future reuse, upload both:

```text
assets/
bluerov_heavy_cad/
```

They should be kept at the same relative locations as in this repository. The
wrapper USD currently depends on the `../../bluerov_heavy_cad/bluerov_heavy.usd`
path, so moving only `assets/` can break the high-detail visual reference.

If the upload target does not preserve symlinks, replace
`bluerov_heavy_cad/bluerov_heavy.usd` with a real copy of
`assets/BlueROVHeavy/bluerov_heavy_fine.usd`, or recreate the symlink after
download.

## Directly Loading the Fine USD

Directly loading `assets/BlueROVHeavy/bluerov_heavy_fine.usd` may show the
detailed CAD visually, but it is not the current simulation path. The existing
Isaac scripts expect the wrapper structure from `BlueROVHeavy.usd`, especially
paths such as:

```text
/World/Vehicle/base_link
/World/Vehicle/base_link/visuals
```

For normal simulation, control, collection, and teleoperation, keep loading
`assets/BlueROVHeavy/BlueROVHeavy.usd`.
