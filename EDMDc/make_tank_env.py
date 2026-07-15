"""Generate ``assets/environment_tank.usda`` — the small physical test tank.

Models the lab water tank (14'9" x 7'3" footprint, 33" walls, partially
filled) as an ASCII USD stage, following the prim layout of the existing
``environment.usd`` 25 m pool so downstream code (GripperScene, collect
scripts) can swap environments via a path only.

Vertical datum matches the Fossen convention: the water surface stays at
``WATER_SURFACE_Z = -0.2`` and the floor drops out of it by the fill
depth, so no physics constant changes anywhere:

    z = +0.04   tank rim              (floor top + wall height)
    z = -0.20   water surface         (Fossen WATER_SURFACE_Z)
    z = -0.80   tank floor top        (surface - water depth)

Floor and walls carry ``PhysicsCollisionAPI`` (static colliders). The
checkerboard floor tiles are visual-only, mirroring
``GripperScene.add_floor_grid``.

Pure pxr — no SimulationApp, so it runs in seconds:

    conda activate marinegym && python -m EDMDc.make_tank_env
    # depth changed? re-generate:
    python -m EDMDc.make_tank_env --water-depth 0.55
"""
from __future__ import annotations

import argparse
from pathlib import Path

FT = 0.3048
IN = 0.0254

TANK_LENGTH_M = 14 * FT + 9 * IN      # 4.4958  (14'9")
TANK_WIDTH_M = 7 * FT + 3 * IN        # 2.2098  (7'3")
TANK_WALL_M = 33 * IN                 # 0.8382  (33")
WATER_SURFACE_Z = -0.2                # Fossen datum — keep in sync with collect scripts
WALL_THICK = 0.10                     # same slab thickness as environment.usd
FLOOR_THICK = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--water-depth", type=float, default=0.6,
                   help="Fill depth in metres, floor top to surface (default 0.6).")
    p.add_argument("--length", type=float, default=TANK_LENGTH_M,
                   help=f"Interior x extent (default {TANK_LENGTH_M:.4f} = 14'9\").")
    p.add_argument("--width", type=float, default=TANK_WIDTH_M,
                   help=f"Interior y extent (default {TANK_WIDTH_M:.4f} = 7'3\").")
    p.add_argument("--wall-height", type=float, default=TANK_WALL_M,
                   help=f"Wall height above floor top (default {TANK_WALL_M:.4f} = 33\").")
    p.add_argument("--tile", type=float, default=0.25,
                   help="Checkerboard tile edge in metres (default 0.25).")
    p.add_argument("--no-checkerboard", action="store_true",
                   help="Skip the visual floor tiles.")
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default assets/environment_tank.usda).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    project = Path(__file__).resolve().parents[1]
    out_path = Path(args.out) if args.out else project / "assets" / "environment_tank.usda"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    L, W = float(args.length), float(args.width)
    depth = float(args.water_depth)
    wall_h = float(args.wall_height)
    floor_top = WATER_SURFACE_Z - depth
    rim_z = floor_top + wall_h
    wall_bot = floor_top - FLOOR_THICK          # flush with the floor slab bottom
    wall_ctr_z = 0.5 * (wall_bot + rim_z)
    wall_ext_z = rim_z - wall_bot

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    env = UsdGeom.Xform.Define(stage, "/Environment")
    stage.SetDefaultPrim(env.GetPrim())

    dome = UsdLux.DomeLight.Define(stage, "/Environment/DomeLight")
    dome.CreateIntensityAttr(500.0)
    sun = UsdLux.DistantLight.Define(stage, "/Environment/SunLight")
    sun.CreateIntensityAttr(1000.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    UsdGeom.Xform.Define(stage, "/Environment/Pool")

    def slab(path: str, center, extent, color) -> None:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        xf.AddScaleOp().Set(Gf.Vec3f(*extent))
        UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    wall_rgb = (0.55, 0.62, 0.68)
    floor_rgb = (0.40, 0.48, 0.55)
    # Floor slab spans to the walls' outer faces, walls sit on its rim
    # (same overlap pattern as environment.usd).
    slab("/Environment/Pool/Floor",
         (0.0, 0.0, floor_top - 0.5 * FLOOR_THICK),
         (L + 2 * WALL_THICK, W + 2 * WALL_THICK, FLOOR_THICK), floor_rgb)
    slab("/Environment/Pool/WallEast",
         (+(L / 2 + WALL_THICK / 2), 0.0, wall_ctr_z),
         (WALL_THICK, W + 2 * WALL_THICK, wall_ext_z), wall_rgb)
    slab("/Environment/Pool/WallWest",
         (-(L / 2 + WALL_THICK / 2), 0.0, wall_ctr_z),
         (WALL_THICK, W + 2 * WALL_THICK, wall_ext_z), wall_rgb)
    slab("/Environment/Pool/WallNorth",
         (0.0, +(W / 2 + WALL_THICK / 2), wall_ctr_z),
         (L, WALL_THICK, wall_ext_z), wall_rgb)
    slab("/Environment/Pool/WallSouth",
         (0.0, -(W / 2 + WALL_THICK / 2), wall_ctr_z),
         (L, WALL_THICK, wall_ext_z), wall_rgb)

    n_tiles = 0
    if not args.no_checkerboard:
        # Visual-only checkerboard (no collision), mirroring
        # GripperScene.add_floor_grid but rectangular and baked in.
        tile = float(args.tile)
        colors = ((0.9, 0.9, 0.9), (0.20, 0.20, 0.20))
        nx, ny = int(L / tile), int(W / tile)
        x0, y0 = -0.5 * nx * tile, -0.5 * ny * tile
        UsdGeom.Xform.Define(stage, "/Environment/FloorGrid")
        for i in range(nx):
            for j in range(ny):
                cube = UsdGeom.Cube.Define(
                    stage, f"/Environment/FloorGrid/tile_{i}_{j}")
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    x0 + (i + 0.5) * tile, y0 + (j + 0.5) * tile,
                    floor_top + 0.006))
                xf.AddScaleOp().Set(Gf.Vec3f(tile, tile, 0.01))
                UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr(
                    [Gf.Vec3f(*colors[(i + j) % 2])])
                n_tiles += 1

    stage.GetRootLayer().Save()

    print(f"[tank-env] wrote {out_path}")
    print(f"[tank-env] interior {L:.4f} x {W:.4f} m, water depth {depth:.3f} m")
    print(f"[tank-env] surface z={WATER_SURFACE_Z:.3f}, floor top z={floor_top:.3f}, "
          f"rim z={rim_z:+.3f}")
    print(f"[tank-env] checkerboard: {n_tiles} tiles @ {args.tile:g} m")


if __name__ == "__main__":
    main()
