from timeit import default_timer as timer
from numba import jit
from stl import mesh
from tqdm import tqdm
import numpy as np
import pathlib
import os

def get_stl_file_list(stl_dir):
    file_list = os.listdir(stl_dir)
    for i, item in reversed(list(enumerate(file_list))):
        file_list[i] = os.path.join(stl_dir,file_list[i])
        if not item.endswith('.stl'):
            file_list.pop(i)
        if item.endswith('.igs'):
            file_list.pop(i)
        if item.endswith('.swl'):
            file_list.pop(i)
    return file_list

# can this be cuda??
@jit(nopython=True)
def is_inside_triangle_plane(triangle, point, strictness = 1):
    # area of triangle = 0.5* ABXAC
    # if both equal, means inside triangle
    # base cannot possibly bigger than parts
    # have to adjust base to make it more insensitive
    if strictness < 1:
        raise ValueError("No possible solution")
    base_triangle = 0.5 * np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))
    return strictness*base_triangle >= (0.5 * np.linalg.norm(np.cross(triangle[2]-point, triangle[0]-point)) + \
                                        0.5 * np.linalg.norm(np.cross(triangle[0]-point, triangle[1]-point)) + \
                                        0.5 * np.linalg.norm(np.cross(triangle[1]-point, triangle[2]-point)))


@jit(nopython=True)
def evaluate_points(item, min_val, max_val, bool_3d):
    for x in range(min_val[0], max_val[0]+1):
        for y in range(min_val[1], max_val[1]+1):
            for z in range(min_val[2], max_val[2]+1):
                # use 3 for 10x smaller, use 1.2 if normal data
                if is_inside_triangle_plane(item, np.array([x,y,z])):
                    if not bool_3d[x,y,z]:
                        bool_3d[x,y,z] = True
    return bool_3d



if __name__ == "__main__":
    file_list = get_stl_file_list('STL')
    target_dir = 'data/stl_bool_data_sample'
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    start = timer()
    for mesh_file in file_list:
        print(f"loading mesh {mesh_file}")

        finished = [os.path.splitext(paths)[0].split('/')[-1].replace(target_dir, "") for paths in os.listdir(target_dir)]
        if mesh_file.split('/')[-1].split('.')[0] in finished:
            print(f"already calculated {mesh_file}")
            continue

        your_mesh = mesh.Mesh.from_file(mesh_file)

        # testing
        your_mesh.v0 = your_mesh.v0
        your_mesh.v1 = your_mesh.v1
        your_mesh.v2 = your_mesh.v2
        # testing

        xmax = np.array([your_mesh.v0[:,0],your_mesh.v1[:,0],your_mesh.v2[:,0]]).max() + 2
        ymax = np.array([your_mesh.v0[:,1],your_mesh.v1[:,1],your_mesh.v2[:,1]]).max() + 2
        zmax = np.array([your_mesh.v0[:,2],your_mesh.v1[:,2],your_mesh.v2[:,2]]).max() + 2

        bool_3d = np.zeros((int(xmax),int(ymax),int(zmax)), dtype=bool)

        for item in tqdm(your_mesh.vectors):
            min_val = np.floor(item.min(axis=0)).astype(int)
            max_val = np.ceil(item.max(axis=0)).astype(int)
            bool_3d = evaluate_points(item,min_val, max_val, bool_3d)

        np.save(f"{target_dir}/{mesh_file[6:-4]}", bool_3d)
        # break

    print(f"time elapsed: {timer()-start}")
