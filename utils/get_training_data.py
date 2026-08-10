import torch
import numpy as np
import matplotlib.pyplot as plt

def remove_duplicate_points(points: torch.Tensor, tol=1e-8):
    """
    Remove duplicate points from a 2D tensor of coordinates.

    Args:
        points (torch.Tensor): shape [N, 2], coordinates.
        tol (float): tolerance for considering two points as equal.

    Returns:
        torch.Tensor: unique points with duplicates removed.
    """
    # Round to tolerance to avoid floating-point issues
    rounded = torch.round(points / tol) * tol
    unique_points = torch.unique(rounded, dim=0)
    return unique_points

def get_data_EX2D1_thermomech_gripper_four_phase_source(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    base_folder = MP['base_folder']
    # 'GPU1' is simply part of the shipped data filenames (kept for
    # compatibility); everything runs on the single device below.
    file_loc = base_folder + 'Data/' + f'{Example}_GPU1_{Diff_type}_thermomechanical.pt'
    mesh_data = torch.load(file_loc)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tkwargs = {'dtype': torch.float32}
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    mask_col = (xi <= 400) | (yi <=150)
    mask_col = mask_col.T.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col[mask_col]
    index = (X_col[:, 0] == xmin) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymin) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= xmax)
    BE = X_col[index] # bottom edge coordinates

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= 400.0)
    tp = X_col[index] # left edge coordinates

    # define the gripper contact
    index = (X_col[:, 0] >= 400.0) & (X_col[:, 0] <= xmax) & (X_col[:, 1] >= 140.0) & (X_col[:, 1] <= 150.0)
    gripper_contact = X_col[index]

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], marker='o', alpha=0.9, s=2, color='green', label = 'bottom edge')
    # ax.scatter(tp[::1,0:1], tp[::1,1:2], marker='o', alpha=0.9, s=2, color='cyan', label = 'top edge')
    # ax.scatter(gripper_contact[::1,0:1], gripper_contact[::1,1:2], marker='o', alpha=0.9, s=2, color='black', label = 'gripper contact')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    LE = torch.tensor(LE)
    BE = torch.tensor(BE)
    tp = torch.tensor(tp)

    u_X_train = torch.cat([LE,BE], dim=0).type(tkwargs["dtype"]).requires_grad_(False)
    u_X_train = remove_duplicate_points(u_X_train)
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"])

    v_X_train = torch.cat([LE,BE,tp], dim=0).type(tkwargs["dtype"]).requires_grad_(False)
    v_X_train = remove_duplicate_points(v_X_train)
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"])

    T_X_train = LE.type(tkwargs["dtype"]).requires_grad_(False)
    T_train = (MP['TD'] * torch.ones_like(T_X_train)[:,0]).type(tkwargs["dtype"])

    # BCs for the adjoint displacement field
    vd1_X_train = u_X_train
    vd1_train = u_train
    vd2_X_train = v_X_train
    vd2_train = v_train
    vt_X_train = T_X_train
    vt_train = torch.zeros_like(T_train).type(tkwargs["dtype"])

    phase0_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase1_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase2_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase3_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase0_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase1_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase2_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase3_train = (torch.ones_like(gripper_contact)[:,0]).type(tkwargs["dtype"])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'T_X_train':T_X_train,'T_train':T_train,
                'vd1_X_train':vd1_X_train,'vd1_train':vd1_train,'vd2_X_train':vd2_X_train,'vd2_train':vd2_train,
                'vt_X_train':vt_X_train,'vt_train':vt_train,
                'phase0_X_train':phase0_X_train,'phase0_train':phase0_train,
                'phase1_X_train':phase1_X_train,'phase1_train':phase1_train,
                'phase2_X_train':phase2_X_train,'phase2_train':phase2_train,
                'phase3_X_train':phase3_X_train,'phase3_train':phase3_train,}
    
    # move the mesh data to the device
    for mesh_key, mesh_dict in mesh_data['GPU0'].items():
        for k, v in mesh_dict.items():
            if k in ["X_node", "X_elem"]:
                # float + gradients
                mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
            elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                # indices must stay long
                mesh_dict[k] = v.to(device, dtype=torch.long)
            else:
                # float but no gradients
                mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data




def get_data_EX2D2_thermomech_actuator_four_phase_source(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    base_folder = MP['base_folder']
    # 'GPU1' is simply part of the shipped data filenames (kept for
    # compatibility); everything runs on the single device below.
    file_loc = base_folder + 'Data/' + f'{Example}_GPU1_{Diff_type}_thermomechanical.pt'
    mesh_data = torch.load(file_loc)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tkwargs = {'dtype': torch.float32}
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    mask_col = (xi <= 500) | (yi <=250)
    mask_col = mask_col.T.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col[mask_col]
    index = (X_col[:, 0] == xmin) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymin) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= xmax)
    BE = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= xmax)
    tp = X_col[index] # left edge coordinates

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], marker='o', alpha=0.9, s=2, color='green', label = 'bottom edge')
    # ax.scatter(tp[::1,0:1], tp[::1,1:2], marker='o', alpha=0.9, s=2, color='cyan', label = 'top edge')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    LE = torch.tensor(LE)
    BE = torch.tensor(BE)
    tp = torch.tensor(tp)

    u_X_train = torch.cat([LE,tp], dim=0).type(tkwargs["dtype"]).requires_grad_(False)
    u_X_train = remove_duplicate_points(u_X_train)
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"])

    v_X_train = torch.cat([BE,LE,tp], dim=0).type(tkwargs["dtype"]).requires_grad_(False)
    v_X_train = remove_duplicate_points(v_X_train)
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"])

    T_X_train = LE.type(tkwargs["dtype"]).requires_grad_(False)
    T_train = (MP['TD'] * torch.ones_like(T_X_train)[:,0]).type(tkwargs["dtype"])

    vd1_X_train = u_X_train
    vd1_train = u_train
    vd2_X_train = v_X_train
    vd2_train = v_train
    vt_X_train = T_X_train
    vt_train = torch.zeros_like(T_train).type(tkwargs["dtype"])

    # define the dummy training set, not used for this example
    phase0_X_train = torch.tensor(tp[0:3]).type(tkwargs["dtype"]).requires_grad_(False)
    phase1_X_train = torch.tensor(tp[0:3]).type(tkwargs["dtype"]).requires_grad_(False)
    phase2_X_train = torch.tensor(tp[0:3]).type(tkwargs["dtype"]).requires_grad_(False)
    phase3_X_train = torch.tensor(tp[0:3]).type(tkwargs["dtype"]).requires_grad_(False)
    phase0_train = (torch.zeros_like(phase0_X_train)[:,0]).type(tkwargs["dtype"])
    phase1_train = (torch.zeros_like(phase1_X_train)[:,0]).type(tkwargs["dtype"])
    phase2_train = (torch.zeros_like(phase2_X_train)[:,0]).type(tkwargs["dtype"])
    phase3_train = (torch.zeros_like(phase3_X_train)[:,0]).type(tkwargs["dtype"])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'T_X_train':T_X_train,'T_train':T_train,
                'vd1_X_train':vd1_X_train,'vd1_train':vd1_train,'vd2_X_train':vd2_X_train,'vd2_train':vd2_train,
                'vt_X_train':vt_X_train,'vt_train':vt_train,
                'phase0_X_train':phase0_X_train,'phase0_train':phase0_train,
                'phase1_X_train':phase1_X_train,'phase1_train':phase1_train,
                'phase2_X_train':phase2_X_train,'phase2_train':phase2_train,
                'phase3_X_train':phase3_X_train,'phase3_train':phase3_train,}
    
    # move the mesh data to the device
    for mesh_key, mesh_dict in mesh_data['GPU0'].items():
        for k, v in mesh_dict.items():
            if k in ["X_node", "X_elem"]:
                # float + gradients
                mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
            elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                # indices must stay long
                mesh_dict[k] = v.to(device, dtype=torch.long)
            else:
                # float but no gradients
                mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data
