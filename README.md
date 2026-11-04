# Creases in the light of fracture mechanics: code and data

This repository holds the code and data behind the paper

> E. Barbieri, *Creases in the light of fracture mechanics*, submitted (2026).

The paper treats a crease that ends inside a thin sheet with the tools of fracture mechanics. The crease is an elastic hinge with a rest angle, in a linear Kirchhoff plate. The code here computes the analytical solutions of the paper, compares them with a nonlinear meshfree code, and rebuilds every figure and table that contains computed results.

## What is in the repository

| Folder | Content |
|---|---|
| `creasefm/` | A small Python package with the analytical solutions: the Glauert solution for the crease (`glauert.py`) and the finite-sheet correction (`finite_sheet.py`). |
| `scripts/` | One script for each figure or table of the paper. Each script writes the data that the figure reads. |
| `data/curvami/` | The processed output of the meshfree code Curvami, used in Section 8 (see `data/curvami/README.md`). |
| `figures/` | The pgfplots and TikZ sources of the figures, their data (`figures/data/`), and the built PDFs (`figures/standalone/`). |

## From the paper to the scripts

| Paper | Script | Run time |
|---|---|---|
| Figure 6: moment near the tip | `scripts/tip_moment.py` | a few minutes |
| Figure 7 and Table 1: fold angle, fold energy release rate | `scripts/rest_angle.py` | under a minute |
| Table 2: path-independent integral, contour and domain forms | `scripts/checks.py 6 7` | 10–30 minutes |
| Figures 10 and 12: finite sheet, path to the edges and size effect | `scripts/finite_sheet_path.py` | 30–60 minutes |
| Figure 11: finite sheet, deflection at four stages | `scripts/finite_sheet_stages.py` | 10–20 minutes |
| Figures 13 and 14: deflection, theory against the code | `scripts/code_deflection.py` | under a minute |
| Figure 15 and Table 3: fold energy release rate, theory against the code | `scripts/code_release_rate.py` | about a minute |
| Table 4: range of the linear theory | `scripts/linear_range.py` | under a minute |
| Appendix D, Table D.6: numerical checks | `scripts/checks.py` | 10–30 minutes |

Figures 1–5, 8 and 9 are diagrams. The sources of Figures 3 and 4 are in `figures/`; the others are drawn in the paper source.

## How to run

You need Python 3.10 or newer with NumPy, SciPy, SymPy and Matplotlib:

```
pip install -r requirements.txt
```

Run one script, or all of them:

```
cd scripts
python rest_angle.py
python run_all.py          # everything, about 1-2 hours
```

To rebuild the figure PDFs from `figures/data/` you need a LaTeX installation with pgfplots and the `txfonts` package:

```
./build_figures.sh               # all figures
./build_figures.sh pgf_fs_Gf     # one figure
```

The data in `figures/data/` and the PDFs in `figures/standalone/` are the ones used in the paper. Running the scripts reproduces them exactly (the images of Figure 6 to within a few pixels).

## Conventions

These follow the paper.

- The crease lies on |x| < a, y = 0, in a plate of bending stiffness D and Poisson's ratio ν.
- The bending moments follow the Timoshenko sign convention: M_xx = −D(w,xx + ν w,yy), M_yy = −D(w,yy + ν w,xx), M_xy = −D(1 − ν) w,xy.
- The fold angle θ(x) is the jump in the slope w,y across the crease. The hinge law is M_yy = −k(θ − θ_r), with k the hinge stiffness per unit length and θ_r the rest angle.
- Λ = D(1 − ν)(3 + ν)/2 is the line stiffness, ℓ_k = Λ/k the origami length, and λ = a/ℓ_k the hinge number.
- U is the stored energy (plate plus hinge), and G_f = −½ dU/da is the fold energy release rate per tip.
- In the finite sheet, |x|, |y| ≤ W, and ξ = kW/D.

## The meshfree code

The nonlinear results of Section 8 come from Curvami, a meshfree code for Kirchhoff–Love plates and shells with creases. Curvami is a separate repository: https://github.com/Ettore-Barbieri/Curvami-Official (DOI: to be added).

## How to cite

Please cite the paper. To cite this code and data, use the Zenodo DOI of this repository (to be added) or the file `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
