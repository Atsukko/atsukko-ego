import torch
from torch import Tensor
from torchmetrics import utilities as ut


def accumulate_as_dict(
    motions: list[Tensor],
    ground_truth: list[Tensor],
    sequences: list[Tensor],
    actions: list[Tensor],
    idxes: list[Tensor],
    skeleton: Tensor,
    use_cache: bool = False,
):
    """
    Accumulate the tensors into a dictionary of sequences, actions, and frames.

    Args:
        motions: list of motions, as predicted by the model. list of tensors of shape (B, J, 3).
        ground_truth: list of ground truth motions. list of tensors of shape (B, J, 3).
        sequences: list of sequences name, encoded as tensors of shape (B, N).
        actions: list of actions name, encoded as tensors of shape (B, N).
        idxes: list of indexes tensor of shape (B, T).
        skeleton: Skeleton data, as a tensor of shape (J, 2).

    Returns:
        A tuple of two dictionaries (all_motions, all_ground_truth), each containing:
            - Outer key: sequence name (str)
            - Inner structure:
                - "skeleton": Tensor of shape (J, 2)
                - "actions": dict[str, Tensor]
                    - Key: action name (str)
                    - Value: Tensor of shape (T, J, 3), where T is the number of frames
    """
    if use_cache:
        return accumulate_as_dict.cache

    sequences = decode_to_string(ut.dim_zero_cat(sequences))  # list of strings of length N
    actions = decode_to_string(ut.dim_zero_cat(actions))  # list of strings of length N
    idxes = ut.dim_zero_cat(idxes)  # Shape: (N, T)
    motions = ut.dim_zero_cat(motions)  # Shape: (N, T, J, 3)
    ground_truth = ut.dim_zero_cat(ground_truth)  # Shape: (N, T, J, 3)

    all_motions = {}
    all_ground_truth = {}

    for seq, act, idxs, jts3D, jts3Dgt in zip(sequences, actions, idxes, motions, ground_truth):
        if seq not in all_motions:
            all_motions[seq] = {"skeleton": skeleton, "actions": {}}
            all_ground_truth[seq] = {"skeleton": skeleton, "actions": {}}

        if act not in all_motions[seq]["actions"]:
            all_motions[seq]["actions"][act] = {}
            all_ground_truth[seq]["actions"][act] = {}

        for idx, jt3D, jt3Dgt in zip(idxs, jts3D, jts3Dgt):
            all_motions[seq]["actions"][act][idx.item()] = jt3D
            all_ground_truth[seq]["actions"][act][idx.item()] = jt3Dgt

    # Flatten the inner dictionaries into single tensors
    for seq in all_motions:
        for act in all_motions[seq]["actions"]:
            idx_to_pose = all_motions[seq]["actions"][act]
            idx_to_gt = all_ground_truth[seq]["actions"][act]

            all_motions[seq]["actions"][act] = torch.stack([idx_to_pose[idx] for idx in sorted(idx_to_pose)])
            all_ground_truth[seq]["actions"][act] = torch.stack([idx_to_gt[idx] for idx in sorted(idx_to_gt)])

    # Save the results in the cache
    accumulate_as_dict.cache = (all_motions, all_ground_truth)
    return all_motions, all_ground_truth


def accumulate_as_tensor(
    motions: list[Tensor],
    ground_truth: list[Tensor],
    sequences: list[Tensor],
    actions: list[Tensor],
    idxes: list[Tensor],
    skeleton: Tensor,
    use_cache: bool = False,
):
    motions, ground_truth = accumulate_as_dict(motions, ground_truth, sequences, actions, idxes, skeleton, use_cache)

    # Flatten the dictionaries into single tensors
    all_motions = []
    all_ground_truth = []
    for seq in motions:
        for act in motions[seq]["actions"]:
            all_motions.append(motions[seq]["actions"][act])
            all_ground_truth.append(ground_truth[seq]["actions"][act])

    all_motions = torch.cat(all_motions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)

    return all_motions, all_ground_truth


def decode_to_string(encoded_strings: list[Tensor] | Tensor) -> list[str]:
    # Convert ASCII values back to strings and remove padding (zeros)
    strings = ["".join(chr(c) for c in lst if c != 0) for lst in encoded_strings]
    return strings
