"""creasefm: analytical solutions for a crease with an elastic hinge in a Kirchhoff plate.

Companion code to E. Barbieri, "Creases in the light of fracture mechanics".
"""
from .glauert import (line_stiffness, glauert_coefficients, theta, moment_on_line,
                      moment_intensity, stored_energy, hinge_energy,
                      fold_energy_release_rate, deflection)
from .finite_sheet import FiniteSheet, winf_derivs
