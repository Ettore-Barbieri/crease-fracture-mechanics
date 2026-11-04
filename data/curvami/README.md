# Output of the meshfree code (Section 8)

These files are the processed output of the meshfree code Curvami
(https://github.com/Ettore-Barbieri/Curvami), build `e573443`. Each file starts with comment lines
(`#`) that give the run and its parameters.

## The runs

- Square sheet [−1, 1]², 80 × 80 cells, Young's modulus E = 9.6 × 10⁷, ν = 0.3.
- Thickness h = 0.005, so D = Eh³/(12(1 − ν²)) = 1.0989010989; two runs use h = 0.01 (files with `_h0.01`).
- Hinge stiffness k = ξD, with ξ = 1, 10 or 100.
- Three corner supports remove the rigid-body motion; they carry no load.
- The crease is centred and extended from a = 0.025 to 0.5 in steps of 0.025.
- Rest angle θ_r: 0.5° (`t0.5`), or 1°, 2°, 5°, 10°, 20° (`t1` … `t20`) for the range of the linear theory.

## The files

| File | Content |
|---|---|
| `CB_n80_t0.5_xi10_a{0.1,0.3,0.5}_field.csv` | Displacements `x, y, u, v, w` at the cell-centre Gauss points (Figures 13 and 14). |
| `CB_n80_t0.5_GD_table.csv` | Fold energy release rate per tip from the code: domain integral on rings ρ₁ < ρ < ρ₂ around each tip (`_s`: analytic weight q; `_r`: interpolated weight), and the energy route (`energy`). Figure 15 and Table 3 use `_s`. |
| `CB_n80_t*_xi*_energy.csv` | Energies at every step: stored energy `Pi` (= U in the paper), bending, membrane and hinge parts (Table 4). |
| `CB_n80_t*_xi*_a{0.1,0.3,0.5}_theta.csv` | Fold angle at the crease Gauss points (Table 4). |
