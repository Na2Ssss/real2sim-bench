# Real2Sim-Bench Cross-Scene Migration Playbook

This guide distills every verified engineering lesson from migrating **248 real DROID episodes** across **3 distinct scenes** (`AUTOLab` cup-in-bowl 100 eps, `AUTOLab` articulated workbench drawer 74 eps, `GuptaLab` motorized standing desk pouring 74 eps) onto **All-on-A100 `MuJoCo 4.0.1`**.

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

---

## 3. Pre-Configured Next DROID Lab Candidates (`PRESET_SCENES`)

All 4 candidates come from the local 24,044-episode calibrated multi-view DROID cache (`data/hf_meta_cache/karlp_droid/`) and already have `SceneTransferConfig` presets ready:

| Preset ID | Lab & Date | Calibrated Eps | Task | Route & Outcome Predicate |
| :--- | :--- | :---: | :--- | :--- |
| `rpl_can_in_bowl` | `RPL 2023-11-30` | 194 | Put can in bowl | Rigid `Hunyuan3D-2mv` + `object_in_container` |
| `tri_marker_from_cup` | `TRI 2023-09-21` | 186 | Remove marker from cup | Slender rigid + hollow cup + `object_out_of_container` |
| `rail_open_drawer` | `RAIL 2023-07-14` | 255 | Open the drawer | Articulated dual-view fit + `joint_state` |
| `iprl_lid_on_pot` | `IPRL 2024-01-01` | 253 | Put lid on pot | Dual rigid `Hunyuan3D-2mv` + `object_on_target` |
