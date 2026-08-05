# Variant Matrix Index

Use these files to run families of configurations.

- `presets/*.yaml`
  - One configuration file per variant name for targeted execution and clearer ownership.
  - Includes all split variants from the matrix presets plus new `qpso` and `apso` entries.

- `all_known_variants_core.yaml`
  - Canonical comparisons only.
  - Core algorithm families: `standard`, `inertia_weight`, `constriction`, `hypersphere`, `fips`, `bare_bones`, `clpso`, `qpso`, `apso`.
  - Neighborhoods: `global`, `ring`, `random`.
  - Boundary defaults: clamping.
  - Additional variant families (non-core): `qpsp`, `qpsb`, `apso_*` topology studies are in `sota_variants.yaml` and `all_known_variants_extended.yaml`.

- `sota_variants.yaml`
  - Full config-first preset matrix for currently implemented SOTA-inspired families.
  - Covers algorithm families and switches that are already implemented in `src/velocity`, `src/influence`, `src/topology`, and `src/boundary`.
  - Retains unsupported combinations as explicit variant intents to document intended behavior under vectorized execution gating.

- `all_known_variants_extended.yaml`
  - Parameter ablations and non-default component configurations.
  - Additional experiments:
    - initialization control (`random_uniform` vs `bounds_aware`)
    - topology density tweaks (`k`, rebuild probability)
    - explicit boundary policy variants (`damped_reflection`, no-velocity-reset clamping)
    - variant-specific velocity tuning (`fast`/`aggressive`/`conservative` presets).
  - Added arXiv-inspired presets:
    - `arxiv_2405_orthogonal_mutation`
    - `arxiv_2503_de_hybrid`
    - `arxiv_2405_orthogonal_learning`

- `all_known_variants.yaml`
  - Curated default preset set (core-aligned convenience launch point).

## Example selectors

```bash
uv run pso-run --config examples/variants/all_known_variants_core.yaml --include-variants standard_ring,inertia_weight,fips_global,clpso
uv run pso-run --config examples/variants/all_known_variants_core.yaml --include-variants qpso,apso
uv run pso-run --config examples/variants/all_known_variants_extended.yaml --include-variants random_sparse,constriction_fixed_chi,clpso_aggressive
uv run pso-run --config examples/variants/all_known_variants_extended.yaml --include-variants qpso_decay,apso_gbest,apso_lbest,apso_stable,apso_aggressive
uv run pso-run --config examples/variants/sota_variants.yaml --include-variants standard_global,orthogonal_learning,de_hybrid,clpso_ring
uv run pso-run --config examples/variants/sota_variants.yaml --include-variants qpsp,qpsb,apso_global_sparse,apso_lbest_dense,apso_random_dense
uv run pso-run --config examples/variants/presets/qpsp.yaml --include-variants qpsp
uv run pso-run --config examples/variants/presets/qpsb.yaml --include-variants qpsb
uv run pso-run --config examples/variants/presets/apso_global_sparse.yaml --include-variants apso_global_sparse
uv run pso-run --config examples/variants/presets/qpso.yaml --include-variants qpso
uv run pso-run --config examples/variants/presets/apso.yaml --include-variants apso
uv run pso-run --config examples/variants/presets/qpso_decay.yaml --include-variants qpso_decay
uv run pso-run --config examples/variants/presets/apso_lbest.yaml --include-variants apso_lbest
```
