# Mast3r GPU SFM Design

This document sketches a small Structure-from-Motion pipeline built on
`wgpu` via the `cubecl` crate.  It is inspired by the Mast3rSfM project and
acts as a starting point for a real implementation.

## Pipeline outline

1. **Feature Extraction** – images are uploaded to the GPU and processed with a
   CubeCL kernel.  The provided example uses `hierarchical_feature_extraction` as
   a placeholder.
2. **Feature Matching** – descriptors are compared in parallel on the GPU.  A
   ratio test is applied on the results to filter matches.
3. **Pose Estimation** – matched features are used to estimate camera poses.
   This stage would implement essential matrix estimation followed by
   incremental pose recovery.
4. **Bundle Adjustment** – once poses and points are available they are refined
   using a GPU‐accelerated optimizer.

The new `Mast3rPipeline` type in `src/mast3r_pipeline.rs` demonstrates how such
an API could look.  Each step currently contains only logging, but real kernels
can be added gradually.

## Known issues in the existing code

- Several modules referenced in `cli.rs` (e.g. `sfm_pipeline`) are missing.
- `nerfstudio.rs` contained unused imports and variables which triggered
  warnings during compilation.
- Many algorithmic kernels in `main.rs` are placeholders and do not implement
  real SfM maths.

## Suggested next steps

1. Replace the placeholder kernels with actual implementations or bindings to
   existing libraries.
2. Flesh out `Mast3rPipeline` so the CLI can call it directly instead of the
   demo code.
3. Add unit tests for the new pipeline to ensure GPU kernels run correctly.

The goal is to evolve this repository into a minimal yet functional GPU
accelerated SfM tool suitable for experimentation with Gaussian splatting and
other modern rendering techniques.
