from __future__ import annotations
import MDAnalysis as mda 
from MDAnalysis.analysis.distances import distance_array
import numpy as np

def spatial_smoothing(
        universe: mda.Universe,
        selection: str,
        cutoff: float):
    selection = u.select_atoms(selection)
    for ts in universe.trajectory:
        distances = distance_array(selection.positions, selection.positions, box=universe.dimensions)
        print(distances)

u = mda.Universe("test/ice_water.gro", "test/ice_water_500.xtc")
spatial_smoothing(u,"type O",10.0)