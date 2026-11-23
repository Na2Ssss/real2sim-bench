# Real2Sim-Bench Cross-Scene Migration Playbook

This guide distills every verified engineering lesson from migrating **608 real DROID episodes** across **10 distinct scenes × 7 independent robotics labs** (`AUTOLab`, `GuptaLab`, `TRI`, `IPRL`, `CLVR`, `PennPAL`, `RPL`) onto **All-on-A100 `MuJoCo 4.0.1`**, raising cross-lab outcome pass rate from **39.5% → 55.8%** (**27.5% → 73.0%** on the 4 zero-shot config-driven scenes).

---

## 1. Onboard Any New Scene in 5 Standard Steps

Every new DROID scene reduces to filling one `SceneTransferConfig` in [`r2s_bench/pipeline/scene_transfer.py`](../r2s_bench/pipeline/scene_transfer.py):

| Stage | Module / Tool | What Runs on A100 | Typical Runtime |
| :--- | :--- | :--- | :---: |
| **1. Data & Hand-Eye** | `droid.py` + `select_camera_extrinsics` | Pull multi-view episode bundle; refine 6-DoF `cam_to_base` against SAM 3.1 robot masks (Robot IoU `0.82–0.93`). | ~20s |
| **2. Segmentation** | `SAM 3.1` (`sam3.1_multiplex.pt`) | Batch-segment Frame-0 Cam1+Cam2 masks for every prompt in `config.assets` + tabletop (`scores.detach().float().cpu().numpy()`). | ~0.6–1.0 s/ep |
| **3. Sim-Ready Assets** | **Route A (`rigid_*`)**: `Hunyuan3D-2mv` + `sim_ready_pipeline.py`<br>**Route B (`articulated_*`)**: Kinematic-anchored Dual-View MuJoCo silhouette fit | • **Rigid**: Gravity de-roll Cam1+Cam2 RGBA cutouts → `Hunyuan3D-2mv` → Z-up norm + radial taper + 3 mm open-top hollow shell + outward normal repair + CoACD (20–24 hulls) + Homography real tabletop texture.<br>• **Articulated**: Anchor axis `yaw` & closed plane from robot FK push/pull vectors → fit **unoccluded protruding part FIRST** on Cam1+Cam2 → lock workbench/cabinet & bake dual-plane Homography textures. | ~60s (once/scene) |
| **4. Zero-Leakage Pose** | `support_mode` router | • `fixed_table`: global `table_z` + Dual-View `[0.75, 0.25]` silhouette XY refine.<br>• `motorized_desk`: pure Frame-0 `triangulate_two_views` recovering per-episode `table_z` (>10 cm desk shifts!).<br>• `articulated_workbench`: closed-linkage Robotiq 2F-85 pad FK (pre-settled at pinch width). | included in rollout |
| **5. Rollout & Eval** | `MuJoCo 4.0.1` (16w `spawn` EGL) + `evaluate_scene_outcome` | Replay episodes in parallel; evaluate **one clean outcome predicate** per task + basic no-NaN/explosion guard. | `0.14–0.39 s/ep` |

---

## 2. Hard-Won Pitfalls & Permanent Fixes (Never Repeat These)

1. **Always use `MuJoCo >= 4.0.1` (never 3.3.2)**:
   - MuJoCo `3.3.2` had a sign-inversion bug on `cam_intrinsic[3]` (`oy`), shifting vertical projection by `+7.8 px` (~`4 cm` world error). Upgrading to `MuJoCo 4.0.1` fixed visual projection to `0.0 px` parity between headless A100 EGL and local renders.
2. **Always use `multiprocessing.get_context("spawn")` for parallel EGL**:
   - Calling Linux default `fork()` after the parent process opens an EGL context (e.g., during Stage A fitting) deadlocks every child worker. `spawn` eliminates this completely.
3. **Never fit an occluded countertop before the unoccluded articulated drawer**:
   - Tabletop clutter (books, trays, bins) makes SAM tabletop masks non-rectangular and drags `yaw` off by `>20°`. Always anchor `yaw` + `closed_plane` from robot push/pull trajectories and fit the **unoccluded protruding drawer/door first**.
4. **Always guard against mid-session camera tripod shifts (`select_camera_extrinsics`)**:
   - In `GuptaLab` (Demo #3), Cam 2's tripod was physically moved `21.7 cm / 14°` partway through the session. Always compare `||t_raw - t_ref|| <= 0.04 m` before applying a global refined extrinsic.
5. **Robotiq 2F-85 4-Bar Linkage `-1.34 cm` Downward Swing**:
   - Closing Robotiq 2F-85 from open (`8.7 cm`) to pinch (`3.8 cm`) swings `"left_silicone_pad"` / `"right_silicone_pad"` (which live on `mjOBJ_BODY`, not `mjOBJ_GEOM`) **downward by `-1.34 cm`**. Always pre-settle `q_fingers_pinch` before running grasp-event FK.
6. **Marching Cubes Inverted Winding & Robotiq Jaw Clearance**:
   - `skimage.measure.marching_cubes` outputs inward faces — always flip `faces[:, ::-1]` + verify signed volume + `fix_normals()`.
   - Robotiq 2F-85 max opening is `8.5 cm`; ensure graspable cups/cans stay `<= 7.4 cm` outer diameter (`target_diameter_m`) for realistic finger clearance.
7. **Outcome-Only Task Predicates (Don't Over-Police Process)**:
   - Requiring artificial process thresholds (`min_lift >= 2 cm`, `upright_cos >= 0.50`, `>= 50%` discrete beans) falsely rejected **23 genuine Demo #2 successes** (humans lift Lego bricks only `0.5–1.5 cm` over a counter edge) and **14 genuine Demo #3 pours**.
   - Use **one clean end-state outcome predicate per task** (`object_in_container`, `object_out_of_container`, `joint_state`, `object_on_target`, `particles_in_target`, `block_in_drawer`) + no-NaN/explosion sanity check.
8. **Genus-1 Containers (Mugs with Side Handles) & Slender Tools (Learned Live on Demo #4 `TRI`)**:
   - CoACD with `max_convex_hull=24` on a genus-1 mug with a side handle bridges across the hollow interior (`concavity=0.127`), ejecting objects placed inside at `t=0`. Always decouple the `Hunyuan3D-2mv` visual mesh (preserving the real handle) from a guaranteed-hollow 16-facet cylindrical cavity collision ring!
   - Official Menagerie `robotiq_2f85` has a **`2.10 cm` minimum closed inner pad gap** at `ctrl=255`. Any slender object (marker/pen/utensil) must have a flat-bottom upright collision cylinder `>= 2.6 cm` diameter (to avoid rolling at `t=0` and to generate real pinch normal force).
   - Always validate SAM 3.1 masks against the workspace grasp region (`<= 14 cm`) to filter background tabletop distractors (e.g., far-end bowls in `TRI`).
9. **Zero-Lag Franka Arm PD (`2.2x` Gain + 1-Frame Lookahead) — Learned Across All 608 Episodes**:
   - Default Menagerie Panda position servo gains lag `2–4` frames (`2–3 cm` spatial lag) behind fast human DROID teleoperation, tipping tall objects (Demo #10 soda can) and stopping `3–5 cm` short of drawer push-end planes (Demo #2).
   - Scaling Franka arm joint `gainprm`/`biasprm` by `2.2x` (damping by `sqrt(2.2)`) plus `1`-frame feedforward lookahead (`qs[min(ti+1, T-1)]`) eliminates open-loop tracking lag with zero instability across all 10 scenes.
10. **Robot Gripper as a Physical Caliper (`8.5 * (1 - g_hold) cm`) & `±axis_grasp` Rim-Pinch Alignment**:
    - Never guess grasped object diameter by eye! The real DROID Robotiq 2F-85 closure plateau `g_hold` directly measures physical grasp thickness (`~2.21 cm` wall on Demo #9 tape roll, `~5.47 cm` on Demo #10 7.5 oz mini soda can).
    - For hollow annular objects (`hollow_ring`) pinched across one wall (`grasp_mode: "rim_pinch"`), offset the ring center strictly along `±axis_grasp` (toward the visual centroid) by `r_mid = r_out - 0.5*wall` and command `hold_ctrl = 0.98` so one finger cleanly enters the hollow center while the other clamps the outer rim — jumping Demo #9 from **`10.0% → 90.0%` (+80 pp)**!
11. **Size-Adaptive Anchor Tolerance (`min(0.28 * min_xy, 0.026)` / `min(0.32 * min_xy, 0.055)`)**:
    - A fixed `6.5 cm` target anchor fallback tolerance is fine for `16 cm` bowls, but catastrophic for precision `4.6 cm` stacking cubes (Demo #7), causing placed cubes to slide off the edge. Tightening tolerance proportionally to object XY footprint jumped Demo #7 stacking from **`22.0% → 84.0%` (+62 pp)**!
12. **Heavy Container Anchoring for In-Container Extraction Tasks**:
    - In Demo #4 (`TRI` marker-from-mug), a lightweight `0.25 kg` mug lifts up together with the marker via inner-wall friction (`13/30` failures). Setting a realistic ceramic mug mass (`0.85 kg`) + low inner-wall friction (`0.20`) keeps the mug planted cleanly while extracting the marker (`50.0% → 56.7%`).

---

## 3. Pre-Configured Next DROID Lab Candidates (`PRESET_SCENES`)

All 4 candidates come from the local 24,044-episode calibrated multi-view DROID cache (`data/hf_meta_cache/karlp_droid/`) and already have `SceneTransferConfig` presets ready:

| Preset ID | Lab & Date | Calibrated Eps | Task | Route & Outcome Predicate |
| :--- | :--- | :---: | :--- | :--- |
| `rpl_can_in_bowl` | `RPL 2023-11-30` | 194 | Put can in bowl | Rigid `Hunyuan3D-2mv` + `object_in_container` |
| `tri_marker_from_cup` | `TRI 2023-09-21` | 186 | Remove marker from cup | Slender rigid + hollow cup + `object_out_of_container` |
| `rail_open_drawer` | `RAIL 2023-07-14` | 255 | Open the drawer | Articulated dual-view fit + `joint_state` |
| `iprl_lid_on_pot` | `IPRL 2024-01-01` | 253 | Put lid on pot | Dual rigid `Hunyuan3D-2mv` + `object_on_target` |
