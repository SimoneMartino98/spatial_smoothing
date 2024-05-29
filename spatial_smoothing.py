from __future__ import annotations
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np

# Main function
def spatial_smoothing(
        universe: mda.Universe,
        array: np.ndarray,
        selection: str,
        cutoff: float) -> np.ndarray:
    selection = universe.select_atoms(selection)
    array = np.load(array)

    if (array.ndim == 2):
        sp_array = np.zeros((array.shape[0], array.shape[1]))
        vector = False
    elif (array.ndim == 3):
        sp_array = np.zeros((array.shape[0], array.shape[1], array.shape[2]))
        vector = True
    else:
        print("INVALID ARRAY SHAPE")

    for ts in universe.trajectory:
        frame = ts.frame
        print(f"FRAME: {frame}")
        distances = distance_array(selection.positions, selection.positions, box=universe.dimensions)
        atom_id = np.argsort(distances, axis=1)
        nn = np.sum(distances < cutoff, axis=1)
        
        rows = np.arange(distances.shape[0])
        sp_dict = {row: atom_id[row, :nn[row]] for row in rows}

        for key, value in sp_dict.items():
            if(vector):
                sp_array[key,frame,:] = np.mean(array[value, frame,:],axis=0)
            else:
                sp_array[key,frame] = np.mean(array[value, frame])

    return sp_array
        

# EXAMPLE
# cutoff = 10
# array = "LENS_10.npy"
# u = mda.Universe("test/ice_water.gro", "test/ice_water_500.xtc")
# sp_array = spatial_smoothing(u, array, "type O", cutoff)
# np.save(f"sp_{cutoff}_{array}", sp_array)

