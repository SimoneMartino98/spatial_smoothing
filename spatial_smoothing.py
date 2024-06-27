import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from multiprocessing import Pool, cpu_count, Array
import ctypes
import time

def init_worker(shared_array, shape, dtype):
    global array
    array = np.frombuffer(shared_array, dtype=dtype).reshape(shape)

def process_frame(args):
    universe, selection, cutoff, frame, vector = args
    universe.trajectory[frame]  # Load the frame explicitly
    distances = distance_array(selection.positions, selection.positions, box=universe.dimensions)
    atom_id = np.argsort(distances, axis=1)
    nn = np.sum(distances < cutoff, axis=1)

    rows = np.arange(distances.shape[0])
    sp_dict = {row: atom_id[row, :nn[row]] for row in rows}

    if vector:
        sp_array_frame = np.zeros((array.shape[0], array.shape[2]))
        for key, value in sp_dict.items():
            if len(value) == 0:
                continue  # Skip if there are no neighbors within cutoff
            sp_array_frame[key, :] = np.mean(array[value, frame, :], axis=0)
    else:
        sp_array_frame = np.zeros(array.shape[0])
        for key, value in sp_dict.items():
            if len(value) == 0:
                continue  # Skip if there are no neighbors within cutoff
            sp_array_frame[key] = np.mean(array[value, frame])
    
    return frame, sp_array_frame

def spatial_smoothing(
        universe: mda.Universe,
        array_path: str,
        selection: str,
        cutoff: float,
        traj_cut: int = 0,
        num_processes: int = None) -> np.ndarray:
    print("CPUs initialization...")
    if num_processes is None:
        num_processes = cpu_count()  
    print("System initialization...")
    selection = universe.select_atoms(selection)
    print("Loading descriptor...")
    array = np.load(array_path)
    
    shape = array.shape
    dtype = array.dtype

    shared_array = Array(ctypes.c_double, array.size, lock=False)
    shared_array_np = np.frombuffer(shared_array, dtype=dtype).reshape(shape)
    np.copyto(shared_array_np, array)
    if array.ndim == 2:
        print(f"  - Monodimensional descriptor of shape: {array.shape}")
        sp_array = np.zeros((array.shape[0], array.shape[1]))
        vector = False
    elif array.ndim == 3:
        print(f"  - Multidimensional descriptor of shape: {array.shape}")
        sp_array = np.zeros((array.shape[0], array.shape[1], array.shape[2]))
        vector = True
    else:
        raise ValueError("INVALID ARRAY SHAPE")

    num_frames = len(universe.trajectory) - traj_cut
    print(f"  - Trajectory length: {num_frames}")
    print("Workers initialization...")
    pool = Pool(processes=num_processes, initializer=init_worker, initargs=(shared_array, shape, dtype))
    print("Collecting spatial average arguments...")
    args = [(universe, selection, cutoff, frame, vector) for frame in range(num_frames)]
    print("Computing spatial average...")
    results = pool.map(process_frame, args)
    pool.close()
    pool.join()
    print("Collecting results...")
    for frame, sp_array_frame in results:
        if vector:
            sp_array[:, frame, :] = sp_array_frame
        else:
            sp_array[:, frame] = sp_array_frame
    print("Process completed")
    return sp_array

# EXAMPLE
cutoff = 10
name = "vel.npy"
array = f"DEV_TEST/{name}"
u = mda.Universe("DEV_TEST/ice_water.gro", "DEV_TEST/ice_water_500.xtc")
sp_array = spatial_smoothing(u, array, "type O", cutoff,0, 8)
print(sp_array)
np.save(f"sp_{cutoff}_p_{name}",sp_array)




