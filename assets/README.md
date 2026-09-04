# Asset Usage Notes

This folder contains the USD and YAML assets used by the Isaac Sim scripts.

## BlueROVHeavy Load Chain

Current code loads the BlueROV2 Heavy model through the small wrapper USD:

```text
EDMDc/*.py
  -> assets/BlueROVHeavy/BlueROVHeavy.usd
      -> ../../bluerov_heavy_cad/bluerov_heavy.usd
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
  - A regular file, currently byte-identical to
    `assets/BlueROVHeavy/bluerov_heavy_fine.usd`, not a symlink.
  - The current simulation loads this copy. Editing only the fine USD copy
    does not change the visual model loaded through the wrapper.

## Download and Reuse

Both directories are versioned and should be kept together:

```text
assets/
bluerov_heavy_cad/
```

Keep them at the same relative locations as in this repository. The
wrapper USD currently depends on the `../../bluerov_heavy_cad/bluerov_heavy.usd`
path, so moving only `assets/` can break the high-detail visual reference.

The two high-detail CAD copies and `BlueROVHeavy/Props/instanceable_meshes.usd`
use **Git Large File Storage (LFS)**. The matching CAD copies share an LFS
content identifier, while both required paths remain available on checkout.
Other small USD/USDA and YAML files use regular Git.

Install Git LFS on the destination machine, then clone normally:

```bash
git lfs install
git clone https://github.com/xrguin/underwater-manipulation-isaac-6.git
cd underwater-manipulation-isaac-6
git lfs pull
git lfs fsck
```

For an existing checkout, run `git pull` followed by `git lfs pull`. If a large
USD opens as a short text file beginning with
`version https://git-lfs.github.com/spec/v1`, it is an LFS pointer rather than
the actual asset: fetch the LFS data before starting Isaac. Prefer an
LFS-enabled clone; do not assume GitHub's source ZIP contains the large assets.
Downloading the assets does not install Isaac Sim or configure its Python
runtime; see the project's `ISAAC6_MIGRATION.md` for those requirements.

### Known dependency limitation

The original `environment.usd` references
`textures/apriltags/tag36h11/tag36_11_00000.png` for its west-wall tag. That PNG
was already absent from the local asset collection and is not included here;
the tag will not have its intended texture. The tank, deep-pool, and pool25
environment files do not reference that missing PNG. `OmniPBR.mdl`, referenced
by the instanceable mesh asset, is supplied by the Isaac Sim installation.

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
