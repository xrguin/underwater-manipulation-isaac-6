"""Visualize and quantify COG vs COB of the BlueROV-Heavy + gripper.

The physical BlueROV is re-trimmed after mounting the gripper (foam moved
so the COB sits back over the COG and the vehicle floats level). The sim
is NOT re-trimmed: Fossen applies vehicle buoyancy at a pure +Z offset
(``coBM``) from the base-link origin, and the gripper adds its own
buoyancy at the nose plus its mass at the nose — the resulting G/B
misalignment is what produces the passive ~17 deg pitch equilibrium.

This script measures the actual composite geometry from PhysX + the
Fossen/GripperScene force bookkeeping:

  * composite COG   (mass-weighted link COMs, PhysX values)
  * effective COB   (buoyancy-force-weighted centroid of the Fossen hull
                     COB and the injected gripper/cube buoyancy points)

places marker spheres on the vehicle (COG yellow, COB blue — at true
position and as side "flags" outside the hull so they are visible), then
settles to equilibrium and CHECKS the story: at rest, B must sit
vertically above G (world-frame horizontal offset ~ 0). Finally it prints
the COB shift a re-trim would need for a level (0 deg) equilibrium.

Run:

    conda activate marinegym && python -m EDMDc.show_cog_cob --headless
    python -m EDMDc.show_cog_cob                 # GUI, orbit the markers
    python -m EDMDc.show_cog_cob --payload-cube  # include the 1 kg cube
"""
from __future__ import annotations

# Argparse FIRST — --help must work without booting Isaac.
import argparse
from pathlib import Path

ENV_FILES = {
    "tank":      "environment_tank.usda",
    "pool":      "environment.usd",
    "deep-pool": "environment_deep_pool.usd",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=sorted(ENV_FILES), default="tank")
    p.add_argument("--payload-cube", action="store_true",
                   help="Weld the 1 kg payload cube and include it in the COG.")
    p.add_argument("--headless", action="store_true",
                   help="No GUI: settle, save screenshots, print, exit.")
    p.add_argument("--centered", action="store_true",
                   help="Shift the composite COG (authored base-link COM) "
                        "and the effective COB (hull COB trim) HORIZONTALLY "
                        "to the base-link origin (x=y=0), preserving the "
                        "vertical metacentric separation BM so passive "
                        "restoring stays. Runs tilt-hold and surge probes "
                        "to confirm stability is retained.")
    p.add_argument("--flag-offset", type=float, default=0.30,
                   help="Sideways (body -y) offset of the visible marker "
                        "flags in metres (default 0.30; 0 disables flags).")
    return p.parse_args()


def main() -> int:
    import numpy as np

    args = parse_args()
    PROJECT = Path(__file__).resolve().parents[1]
    env_usd = str(PROJECT / "assets" / ENV_FILES[args.env])
    out_dir = PROJECT / "EDMDc" / "data" / "plots" / "cog_cob"
    out_dir.mkdir(parents=True, exist_ok=True)

    from .isaac_scene import (
        GripperScene, look_at_quat_wxyz,
        GRIPPER_OFFSET, GRIPPER_BUOY_N, CUBE_OFFSET,
    )
    from .fossen import RHO_WATER, GRAVITY

    scene = GripperScene(dt=1 / 60, headless=args.headless, env_usd=env_usd,
                         payload_cube=args.payload_cube)

    # ----- extra views to read per-link mass + COM from PhysX ---------------
    from omni.isaac.core.prims import RigidPrimView

    grip_view = RigidPrimView(prim_paths_expr="/World/Gripper/base_link",
                              name="grip_com_view")
    scene.world.scene.add(grip_view)
    cube_view = None
    if args.payload_cube:
        cube_view = RigidPrimView(prim_paths_expr="/World/Cube/base_link",
                                  name="cube_com_view")
        scene.world.scene.add(cube_view)
    scene.world.reset()

    def link_mass_com_world(view):
        """(mass, world COM) of a single-prim RigidPrimView."""
        m = float(view.get_masses()[0])
        pos, quat = view.get_world_poses()
        R = scene._quat_to_rotmat(np.asarray(quat[0], dtype=float))
        try:
            com_local, _ = view.get_coms()
            com_local = np.asarray(com_local, dtype=float).reshape(-1)[:3]
        except Exception:
            com_local = np.zeros(3)
        return m, np.asarray(pos[0], dtype=float) + R @ com_local

    def body_frame(p_world, base_pos, R_base):
        return R_base.T @ (np.asarray(p_world, dtype=float) - base_pos)

    def composite_geometry():
        """Return dict with COG/COB in world and vehicle-body frames."""
        base_pos, base_quat = scene.base_view.get_world_poses()
        base_pos = np.asarray(base_pos[0], dtype=float)
        R = scene._quat_to_rotmat(np.asarray(base_quat[0], dtype=float))

        # --- COG: mass-weighted PhysX link COMs --------------------------
        links = [link_mass_com_world(scene.base_view),
                 link_mass_com_world(grip_view)]
        if cube_view is not None:
            links.append(link_mass_com_world(cube_view))
        m_tot = sum(m for m, _ in links)
        cog_w = sum(m * c for m, c in links) / m_tot

        # --- effective COB: buoyancy-weighted force application points ---
        # Vehicle hull: rho*g*V at base + R@(0,0,coBM)  (Fossen)
        # Gripper: GRIPPER_BUOY_N at base + R@GRIPPER_OFFSET (GripperScene)
        # Cube (optional): cube buoyancy at base + R@CUBE_OFFSET
        B_hull = RHO_WATER * GRAVITY * scene._fossen.p.volume
        pts = [(B_hull, base_pos + R @ np.array([scene._fossen.p.cob_x,
                                                 scene._fossen.p.cob_y,
                                                 scene._fossen.p.cob_offset])),
               (GRIPPER_BUOY_N, base_pos + R @ GRIPPER_OFFSET)]
        if args.payload_cube:
            pts.append((scene._cube_buoy_N, base_pos + R @ CUBE_OFFSET))
        B_tot = sum(b for b, _ in pts)
        cob_w = sum(b * p for b, p in pts) / B_tot

        return {
            "m_tot": m_tot, "B_tot": B_tot,
            "cog_w": cog_w, "cob_w": cob_w,
            "cog_b": body_frame(cog_w, base_pos, R),
            "cob_b": body_frame(cob_w, base_pos, R),
            "base_pos": base_pos, "R": R,
        }

    # ----- optional: center COG and COB horizontally (keep BM) --------------
    if args.centered:
        from pxr import UsdPhysics, Gf as _Gf

        pre = composite_geometry()
        # Authored base-link COM: cancel the composite x/y mass moment so
        # the composite COG sits at x=y=0. The z moment is left untouched
        # (heights unchanged -> metacentric separation preserved).
        m_b = float(scene.base_view.get_masses()[0])
        moment = pre["m_tot"] * pre["cog_b"]        # total m*r about origin
        moment[2] = 0.0                             # horizontal shift only
        _, base_com_w = link_mass_com_world(scene.base_view)
        r_b = body_frame(base_com_w, pre["base_pos"], pre["R"])
        r_b_new = r_b - moment / m_b
        UsdPhysics.MassAPI(
            scene.stage.GetPrimAtPath("/World/Vehicle/base_link")
        ).CreateCenterOfMassAttr().Set(_Gf.Vec3f(*[float(v) for v in r_b_new]))

        # Hull COB x/y: cancel the horizontal buoyancy moment of the
        # nose-mounted points (gripper foam, optional cube) so the
        # effective COB is at x=y=0. cob_offset (z) stays -> BM stays.
        B_hull = RHO_WATER * GRAVITY * scene._fossen.p.volume
        others = [(GRIPPER_BUOY_N, GRIPPER_OFFSET)]
        if args.payload_cube:
            others.append((scene._cube_buoy_N, CUBE_OFFSET))
        b_moment = sum(b * np.asarray(r, dtype=float) for b, r in others)
        scene._fossen.p.cob_x = float(-b_moment[0] / B_hull)
        scene._fossen.p.cob_y = float(-b_moment[1] / B_hull)
        print(f"[centered] authored base COM -> {np.round(r_b_new, 5)} m")
        print(f"[centered] hull COB          -> "
              f"({scene._fossen.p.cob_x:+.5f}, {scene._fossen.p.cob_y:+.5f}, "
              f"{scene._fossen.p.cob_offset:+.5f}) m")
        scene.world.reset()     # PhysX re-reads authored mass properties

    # ----- settle to the passive equilibrium --------------------------------
    state = scene.settle(max_s=12.0, min_s=6.0)
    geo = composite_geometry()

    # ----- markers: true-position spheres + visible side flags --------------
    from pxr import UsdGeom, Gf

    def sphere(path, body_pos, rgb, radius=0.022):
        sp = UsdGeom.Sphere.Define(scene.stage, path)
        sp.CreateRadiusAttr(radius)
        xf = UsdGeom.Xformable(sp.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in body_pos]))
        UsdGeom.Gprim(sp.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def rod(path, p0, p1, rgb, radius=0.0035, extend=0.10):
        """Thin cylinder through p0->p1, extended past both ends."""
        d = np.asarray(p1, float) - np.asarray(p0, float)
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return
        dhat = d / L
        mid = 0.5 * (np.asarray(p0, float) + np.asarray(p1, float))
        cyl = UsdGeom.Cylinder.Define(scene.stage, path)
        cyl.CreateRadiusAttr(radius)
        cyl.CreateHeightAttr(L + 2 * extend)
        cyl.CreateAxisAttr("Z")
        # quaternion rotating +Z onto dhat
        z = np.array([0.0, 0.0, 1.0])
        c = float(np.dot(z, dhat))
        axis = np.cross(z, dhat)
        s = float(np.linalg.norm(axis))
        if s < 1e-9:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = axis / s
            half = 0.5 * np.arctan2(s, c)
            quat = np.array([np.cos(half), *(np.sin(half) * axis)])
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in mid]))
        xf.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat]))
        UsdGeom.Gprim(cyl.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    YELLOW, BLUE, WHITE = (1.0, 0.85, 0.0), (0.1, 0.35, 1.0), (0.95, 0.95, 0.95)
    root = "/World/Vehicle/base_link"          # welded frame: markers ride along
    sphere(f"{root}/cog_marker", geo["cog_b"], YELLOW, radius=0.008)
    sphere(f"{root}/cob_marker", geo["cob_b"], BLUE, radius=0.008)
    if args.flag_offset > 0.0:
        off = np.array([0.0, -args.flag_offset, 0.0])
        sphere(f"{root}/cog_flag", geo["cog_b"] + off, YELLOW, radius=0.012)
        sphere(f"{root}/cob_flag", geo["cob_b"] + off, BLUE, radius=0.012)
        # B-G axis rod: vertical in the world exactly at equilibrium, so its
        # tilt against the wall/floor lines shows the residual directly.
        rod(f"{root}/bg_axis", geo["cog_b"] + off, geo["cob_b"] + off, WHITE)

    # ----- report ------------------------------------------------------------
    np.set_printoptions(precision=4, suppress=True)
    d_b = geo["cob_b"] - geo["cog_b"]
    d_w = geo["cob_w"] - geo["cog_w"]
    pitch_deg = float(np.rad2deg(state.pitch))

    print("\n================= COG / COB REPORT =================")
    print(f"composite mass      : {geo['m_tot']:.3f} kg "
          f"(weight {geo['m_tot'] * GRAVITY:.2f} N)")
    print(f"total buoyancy      : {geo['B_tot']:.2f} N")
    print(f"COG  (body frame)   : {geo['cog_b']} m   <- yellow marker")
    print(f"COB  (body frame)   : {geo['cob_b']} m   <- blue marker")
    print(f"B-G  (body frame)   : {d_b} m")
    print(f"  BG_x (fore-aft)   : {d_b[0] * 1000:+.1f} mm "
          f"(negative = COB aft of COG)")
    print(f"  BG_z (metacentric): {d_b[2] * 1000:+.1f} mm")
    print(f"settled pitch       : {pitch_deg:+.2f} deg "
          f"(roll {np.rad2deg(state.roll):+.2f} deg)")
    print(f"predicted eq. pitch : {np.rad2deg(np.arctan2(-d_b[0], d_b[2])):+.2f} deg "
          f"(from atan2(-BG_x, BG_z) — B must end up above G)")
    print("---- consistency check at equilibrium (world frame) ----")
    print(f"horizontal B-G offset: ({d_w[0] * 1000:+.2f}, {d_w[1] * 1000:+.2f}) mm "
          f"(should be ~0 if the marker points are the effective ones)")
    print(f"residual pitch moment: {abs(d_w[0]) * geo['B_tot']:.4f} N*m")
    # Re-trim: shift the HULL COB forward so the effective COB x matches
    # the COG x at zero pitch (what moving the foam does on the real ROV).
    B_hull = RHO_WATER * GRAVITY * scene._fossen.p.volume
    x_shift = (geo["B_tot"] * geo["cog_b"][0]
               - (geo["B_tot"] * geo["cob_b"][0])) / B_hull + 0.0
    print("---- re-trim to float level (real-ROV foam move) ----")
    print(f"hull COB fwd shift  : {x_shift * 1000:+.1f} mm "
          f"(requires cob_offset as a 3-vector in fossen.py; "
          f"currently z-only coBM={scene._fossen.p.cob_offset} m)")

    # ----- screenshots / GUI -------------------------------------------------
    from omni.isaac.sensor import Camera

    def make_cam(path, pos, target, res=(1280, 720), focal=24.0):
        prim = UsdGeom.Camera.Define(scene.stage, path)
        prim.CreateFocalLengthAttr(focal)
        prim.CreateHorizontalApertureAttr(20.955)
        prim.CreateVerticalApertureAttr(20.955 * res[1] / res[0])
        prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        xf = UsdGeom.Xformable(prim.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
        q = look_at_quat_wxyz(np.asarray(pos, float), np.asarray(target, float))
        xf.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in q]))
        cam = Camera(prim_path=path, resolution=res)
        cam.initialize()
        return cam

    # Camera must stay INSIDE the tank (south wall interior face y=-1.105).
    base = geo["base_pos"]
    cam_side = make_cam("/World/cam_side", base + np.array([0.0, -0.95, 0.25]),
                        base, focal=16.0)
    cam_close = make_cam("/World/cam_close", base + np.array([0.35, -0.85, 0.05]),
                         base + np.array([0.05, 0.0, -0.02]))

    zeros6 = np.zeros(6)
    scene.force_render = True
    for _ in range(40):                      # let the ray tracer converge
        scene.apply_wrench(zeros6, render=True)

    import cv2
    tag = "cube" if args.payload_cube else "gripper"
    for name, cam in (("side", cam_side), ("close", cam_close)):
        rgba = cam.get_rgba()
        if rgba is not None and rgba.size:
            out = out_dir / f"cog_cob_{tag}_{name}.png"
            cv2.imwrite(str(out),
                        np.ascontiguousarray(rgba[..., :3][..., ::-1]))
            print(f"screenshot -> {out}")

    if args.centered:
        print("\n---- attitude-neutrality probes (expected consequence of BM=0) ----")
        # Probe 1: tilt-hold. Pitch 10 deg, release. Stable vehicle returns
        # to equilibrium; neutral vehicle holds the tilt indefinitely.
        q10 = np.array([np.cos(np.deg2rad(5.0)), 0.0, np.sin(np.deg2rad(5.0)), 0.0])
        scene.base_view.set_world_poses(
            positions=geo["base_pos"].reshape(1, 3),
            orientations=q10.reshape(1, 4))
        scene.base_view.set_velocities(velocities=np.zeros((1, 6)))
        for _ in range(int(round(5.0 / scene.dt))):
            scene.apply_wrench(zeros6)
        st = scene.read_state()
        print(f"tilt-hold: released at +10.0 deg pitch, after 5 s -> "
              f"{np.rad2deg(st.pitch):+.2f} deg "
              f"(neutral = stays ~10, stable = returns ~0)")
        # Probe 2: surge drift. Level start, moderate surge; the vectored
        # horizontals sit below the COG, so their residual pitch torque now
        # acts on a vehicle with zero restoring stiffness.
        scene.reset_to_spawn()
        scene.settle(max_s=5.0, min_s=3.0)
        print("surge drift @ cmd 0.3:")
        for k in range(int(round(5.0 / scene.dt))):
            scene.apply_wrench(np.array([0.3, 0.0, 0.0, 0.0]))
            if k % 60 == 59:
                st = scene.read_state()
                print(f"  t={(k + 1) * scene.dt:3.1f}s  "
                      f"pitch={np.rad2deg(st.pitch):+7.2f} deg  "
                      f"u={st.nu[0]:+.3f} m/s")

    if not args.headless:
        print("[cog-cob] GUI hold — orbit the viewport; Ctrl-C or close to exit.")
        try:
            while scene.simulation_app.is_running():
                scene.apply_wrench(zeros6)
        except KeyboardInterrupt:
            pass

    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
