import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations


def get_cartpole_state_group_representations():
    """
    Representation of the group symmetry on the state: a multiplication of all
    state variables by -1
    """
    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(-1 * np.eye(4))]
    return GroupRepresentations(representations, "CartPoleStateGroupRepr")


def get_lunar_state_group_representations():
    """
    Representation of the group symmetry on the state: a multiplication of all
    state variables by -1
    """
    representations = [torch.FloatTensor(np.eye(8)),
                       torch.FloatTensor([
                           [-1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, -1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, -1, 0, 0, 0],
                           [0, 0, 0, 0, 0, -1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 1, 0]
                       ])]
    return GroupRepresentations(representations, "LunarStateGroupRepr")


def get_cartpole_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    representations = [torch.FloatTensor(np.eye(2)),
                       torch.FloatTensor(np.array([[0, 1], [1, 0]]))]
    return GroupRepresentations(representations, "CartPoleActionGroupRepr")


def get_lunar_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor([[1, 0, 0, 0],
                                          [0, 0, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 1, 0, 0]])]
    return GroupRepresentations(representations, "LunarActionGroupRepr")


def get_cartpole_invariants():
    """
    Function to enable easy construction of invariant layers (for value
    networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "CartPoleInvariantGroupRepr")


def get_lunar_invariants():
    """
    Function to enable easy construction of invariant layers (for value
    networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "LunarInvariantGroupRepr")


def get_pvz_invariants():
    """
    Function to enable easy construction of invariant layers (for value
    networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "LunarInvariantGroupRepr")


def get_pvz_state_group_representations():
    representations = get_in_mat()
    return GroupRepresentations(representations, "LunarStateGroupRepr")


def get_pvz_action_group_representations():
    representations = get_out_mat()
    return GroupRepresentations(representations, "LunarActionGroupRepr")


def get_in_mat():
    a1 = torch.eye(6)
    a0 = torch.zeros((6, 6))
    b1 = torch.eye(3)
    c95 = torch.zeros((6, 3))
    c59 = torch.zeros((3, 6))
    r0 = torch.cat((a1, a0, a0, a0, a0, a0, c95), dim=1)
    r1 = torch.cat((a0, a1, a0, a0, a0, a0, c95), dim=1)
    r2 = torch.cat((a0, a0, a1, a0, a0, a0, c95), dim=1)
    r3 = torch.cat((a0, a0, a0, a1, a0, a0, c95), dim=1)
    r4 = torch.cat((a0, a0, a0, a0, a1, a0, c95), dim=1)
    r5 = torch.cat((a0, a0, a0, a0, a0, a1, c95), dim=1)
    r6 = torch.cat((c59, c59, c59, c59, c59, c59, b1), dim=1)

    trans_iden = torch.cat((r0, r1, r2, r3, r4, r5, r6), dim=0)
    trans_01 = torch.cat((r1, r0, r2, r4, r3, r5, r6), dim=0)
    trans_02 = torch.cat((r2, r1, r0, r5, r4, r3, r6), dim=0)
    trans_12 = torch.cat((r0, r2, r1, r3, r5, r4, r6), dim=0)

    assert trans_iden.shape == trans_01.shape == trans_02.shape == trans_02.shape == trans_12.shape == (39, 39)

    return [trans_iden.float(), trans_01.float(), trans_02.float(), trans_12.float()]

def get_out_mat():
    iden = torch.eye(37)
    res = [iden[0:1]]

    for i in range(1, 37, 2):
        res.append(iden[i: i + 2])

    res01 = res.copy()

    for i in range(1, 18, 2):
        res01[i], res01[i + 1] = res01[i + 1], res01[i]

    trans_01 = torch.cat(res01, dim=0)

    res02 = res.copy()

    for i in range(1, 18, 3):
        res02[i], res02[i + 2] = res02[i + 2], res02[i]

    trans_02 = torch.cat(res02, dim=0)

    res12 = res.copy()

    for i in range(1, 18, 3):
        res12[i + 1], res12[i + 2] = res12[i + 2], res12[i + 1]

    trans_12 = torch.cat(res12, dim=0)

    return [torch.eye(37).float(), trans_01.float(), trans_02.float(), trans_12.float()]
