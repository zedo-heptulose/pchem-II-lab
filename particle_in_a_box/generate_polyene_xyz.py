"""
Generate all-trans polyene XYZ files with idealized bond lengths.

All-trans conjugated polyenes are built in the xy-plane with:
  - C=C double bonds: 1.34 Angstrom
  - C-C single bonds: 1.46 Angstrom
  - C-H bonds: 1.08 Angstrom
  - All bond angles: 120 degrees (sp2)

The carbon backbone zigzags in the xy-plane. All atoms are planar.

Usage:
    python generate_polyene_xyz.py

Outputs:
    data/ethane.xyz
    data/butadiene.xyz
    data/hexatriene.xyz
    data/octatetraene.xyz
    data/decapentaene.xyz
"""

import numpy as np
import os

# Bond lengths in Angstroms
CC_DOUBLE = 1.34
CC_SINGLE = 1.46
CH_BOND = 1.08
BOND_ANGLE = np.radians(120.0)


def build_polyene(n_carbons):
    """
    Build an all-trans polyene with n_carbons carbon atoms.

    Parameters
    ----------
    n_carbons : int
        Number of carbon atoms (must be even, >= 4).

    Returns
    -------
    atoms : list of (str, float, float, float)
        List of (element, x, y, z) tuples.
    """
    # Build carbon backbone zigzagging in the xy-plane.
    # Bond 0 (C0-C1) is a double bond along +x.
    # At each carbon, the next bond turns by +/- (pi - 120°) = +/- 60°,
    # alternating sign for the all-trans configuration.
    carbons = [(0.0, 0.0)]
    direction = 0.0  # angle of current bond

    for i in range(n_carbons - 1):
        bond_len = CC_DOUBLE if i % 2 == 0 else CC_SINGLE
        x_prev, y_prev = carbons[-1]
        x_new = x_prev + bond_len * np.cos(direction)
        y_new = y_prev + bond_len * np.sin(direction)
        carbons.append((x_new, y_new))

        # Turn for the next bond: alternate sign for trans
        turn = np.pi - BOND_ANGLE  # 60 degrees
        if i % 2 == 0:
            direction -= turn
        else:
            direction += turn

    atoms = [('C', x, y, 0.0) for x, y in carbons]

    # Add hydrogens
    for i in range(n_carbons):
        x_c, y_c = carbons[i]

        if i == 0:
            # Terminal carbon: two H atoms at 120° from the C-C bond
            bond_dir = np.arctan2(
                carbons[1][1] - y_c, carbons[1][0] - x_c
            )
            for sign in [1, -1]:
                h_angle = bond_dir + np.pi + sign * (np.pi - BOND_ANGLE)
                hx = x_c + CH_BOND * np.cos(h_angle)
                hy = y_c + CH_BOND * np.sin(h_angle)
                atoms.append(('H', hx, hy, 0.0))

        elif i == n_carbons - 1:
            # Terminal carbon: two H atoms at 120° from the C-C bond
            bond_dir = np.arctan2(
                carbons[i - 1][1] - y_c, carbons[i - 1][0] - x_c
            )
            for sign in [1, -1]:
                h_angle = bond_dir + np.pi + sign * (np.pi - BOND_ANGLE)
                hx = x_c + CH_BOND * np.cos(h_angle)
                hy = y_c + CH_BOND * np.sin(h_angle)
                atoms.append(('H', hx, hy, 0.0))

        else:
            # Internal sp2 carbon: one H atom opposite the bisector
            dx1 = carbons[i - 1][0] - x_c
            dy1 = carbons[i - 1][1] - y_c
            dx2 = carbons[i + 1][0] - x_c
            dy2 = carbons[i + 1][1] - y_c

            r1 = np.hypot(dx1, dy1)
            r2 = np.hypot(dx2, dy2)
            bis_x = dx1 / r1 + dx2 / r2
            bis_y = dy1 / r1 + dy2 / r2
            r_bis = np.hypot(bis_x, bis_y)

            # H goes opposite the bisector (away from the two C-C bonds)
            hx = x_c - CH_BOND * bis_x / r_bis
            hy = y_c - CH_BOND * bis_y / r_bis
            atoms.append(('H', hx, hy, 0.0))

    return atoms


def build_ethane():
    """
    Build a staggered ethane molecule with idealized geometry.

    Returns
    -------
    atoms : list of (str, float, float, float)
    """
    cc = 1.54
    ch = 1.09
    # Tetrahedral angle from bond axis
    tet = np.radians(109.47)
    half_cc = cc / 2.0

    atoms = [('C', -half_cc, 0.0, 0.0), ('C', half_cc, 0.0, 0.0)]

    # H on C1 (at -half_cc): pointing in -x direction
    h_axial = ch * np.cos(np.pi - tet)  # projection along -x
    h_perp = ch * np.sin(np.pi - tet)   # perpendicular distance
    for k in range(3):
        angle = 2 * np.pi * k / 3
        atoms.append(('H',
                       -half_cc - h_axial,
                       h_perp * np.cos(angle),
                       h_perp * np.sin(angle)))

    # H on C2 (at +half_cc): pointing in +x direction, staggered 60°
    for k in range(3):
        angle = 2 * np.pi * k / 3 + np.pi / 3
        atoms.append(('H',
                       half_cc + h_axial,
                       h_perp * np.cos(angle),
                       h_perp * np.sin(angle)))

    return atoms


def write_xyz(atoms, filename, comment=""):
    """Write atoms to an XYZ file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for elem, x, y, z in atoms:
            f.write(f"{elem:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def main():
    os.makedirs('data', exist_ok=True)

    molecules = [
        ('ethane', 2, build_ethane),
        ('butadiene', 4, None),
        ('hexatriene', 6, None),
        ('octatetraene', 8, None),
        ('decapentaene', 10, None),
    ]

    for name, n_c, builder in molecules:
        atoms = builder() if builder else build_polyene(n_c)
        filename = os.path.join('data', f'{name}.xyz')
        comment = f"all-trans {name}, idealized geometry"
        write_xyz(atoms, filename, comment)
        print(f"Wrote {filename}: {len(atoms)} atoms")


if __name__ == '__main__':
    main()
