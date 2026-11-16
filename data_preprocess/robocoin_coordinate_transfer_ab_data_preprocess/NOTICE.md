# Third-party URDF notice

`urdf/models/` contains URDF files collected from public robot-description
repositories. They are included only because the A1/B1 builder needs the
kinematic trees for forward kinematics.

Source URLs, commits and licenses are recorded in
[`docs/URDF_SOURCES.md`](docs/URDF_SOURCES.md). Keep that file with the URDFs
when redistributing this package.

If you replace or add a URDF, update `urdf/validation/calibration.json` and
record the new source/license in `docs/URDF_SOURCES.md`.
