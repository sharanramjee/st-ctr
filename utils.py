import torch
import scipy.io
import numpy as np
from torch_geometric.data import Data


def load_mat_file(file_path):
    content = scipy.io.loadmat(file_path)
    return content['pos_img']


def generate_graph(mat_content):
    edge_index = torch.tensor(
        [[2, 0], [0, 2], [0, 1], [1, 0],         # torso
         [3, 7], [7, 3], [7, 11], [11, 7],       # right arm
         [4, 8], [8, 4], [8, 12], [12, 8],       # left arm
         [5, 9], [9, 5], [9, 13], [13, 9],       # right leg
         [6, 10], [10, 6], [10, 14], [14, 10],   # left leg
         [0, 3], [3, 0], [0, 4], [4, 0],         # upper connections
         [1, 5], [5, 1], [1, 6], [6, 1],         # lower connections
         [2, 15], [0, 15], [1, 15],              # torso node
         [3, 16], [7, 16], [11, 16],             # right arm node
         [4, 17], [8, 17], [12, 17],             # left arm node
         [5, 18], [9, 18], [13, 18],             # right leg node
         [6, 19], [10, 19], [14, 19],            # left leg node
         [15, 20], [16, 20], [17, 20], [18, 20], [19, 20]],   # skeleton node
        dtype=torch.long)
    graphs = list()
    for frame in range(mat_content.shape[2]):
        feats = mat_content[:, :, frame].T
        print(feats.shape)
        feats = np.vstack((feats, [np.mean(feats[0: 3])]))
        print(feats.shape)
        # graphs.append(Data(x=torch.Tensor(mat_content[:, :])))
    # x = torch.Tensor(mat_content[:, :])

if __name__ == '__main__':
    matfile_path = 'data/joint_positions/walk/50_FIRST_DATES_walk_f_cm_np1_le_med_33/joint_positions.mat'
    matfile_content = load_mat_file(matfile_path)
    generate_graph(matfile_content)
