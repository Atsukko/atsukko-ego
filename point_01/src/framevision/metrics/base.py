from pathlib import Path

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics import utilities as ut

from framevision import geometry as geo
from framevision.dataset.keypoint_map import KEYPOINT_NAMES
from framevision.metrics.utils import accumulate as ac


class MotionMetric(Metric):
    """Base class for sequence-wide motion metrics.
    It's a bit chaotic, but exposes useful methods to subclasses.

    It collects motion predictions and ground truth for each sequence and action.
    Each subclass should implement the `.compute()` method to calculate the metric.

    Methods:
        .update(prediction: dict, batch: dict) Updates the metric with a batch of data.
        .compute() Computes the metric. To be implemented by subclasses.
        .plot() Plots the metric. To be optionally implemented by subclasses.
        .accumulate(shape) Returns the accumulated motion predictions and ground truth in a dictionary or tensor. Shape can be "dict" or "flat".

    During your .compute() implementation, you can access the following attributes:

    Attributes:
        sequences: list of sequence names, encoded as tensors. Shape: (N, S), where N is the number of sequence-action pairs and S is the maximum sequence length.
        actions: list of action names, encoded as tensors. Shape: (N, A), where A is the maximum action length.
        indexes: list of frame indexes. Shape: (N, T), where T is the number of frames.
        skeletons: list of skeleton data tensors. Shape: (N, J, 3), where J is the number of joints.
        motions: list of motion data tensors. Shape: (N, T, J, 3), where T is the number of frames predicted, usually 1.
        ground_truth: list of ground truth data tensors. Shape: (N, T, J, 3), where T is the number of frames predicted, usually 1.
    """

    _use_cache: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sequences", [], dist_reduce_fx="cat")
        self.add_state("actions", [], dist_reduce_fx="cat")
        self.add_state("indexes", [], dist_reduce_fx="cat")
        self.add_state("motions", [], dist_reduce_fx="cat")
        self.add_state("ground_truth", [], dist_reduce_fx="cat")
        self.skeleton_idxes = None
        self.skeleton = None

    def update(self, prediction: dict, batch: dict):
        sequences = batch["meta"]["sequence"]  # Shape: (B,)
        actions = batch["meta"]["action"]  # Shape: (B,)
        idxes = batch["meta"]["idx"]  # Shape: (B, T)
        ground_truth = batch["body_tracking"]["joints_3D"]  # Shape: (B, T, J, 3)
        joints3Dwr = self.get_world_prediction(prediction, batch)  # Shape: (B, T, J, 3)

        # if only the last  step is predicted, we take only the last step of the ground truth
        if joints3Dwr.shape[1] == 1 and ground_truth.shape[1] > 1:
            ground_truth = ground_truth[:, -1:]

        self.sequences.append(encode_strings(sequences))
        self.actions.append(encode_strings(actions))
        self.indexes.append(idxes.detach().cpu())
        self.ground_truth.append(ground_truth.detach().cpu().half())

        if self.skeleton_idxes is None:
            self.skeleton_idxes = batch["meta"]["skeleton_idxes"].detach().cpu()
        if self.skeleton is None:
            self.skeleton = batch["skeleton"][0].detach().cpu()

        self.motions.append(joints3Dwr.half())

        self._use_cache = False

    def compute(self):
        raise NotImplementedError

    def plot(self):
        return None

    def accumulate(self, shape: str = "dict"):
        """Helper function to accumulate the quantities collected during the update() method.

        If shape is "dict", it returns a tuple of two dictionary with the following structure:
        {
            sequence_name: {
                skeleton: Tensor of shape (J, 3),
                actions: {
                    action_name: Tensor of shape (T, J, 3)
                }
            }
        }

        If shape is "flat", it returns a tuple of tensors
        (
            motions: Tensor of shape (N, J, 3),
            ground_truth: Tensor of shape (N, J, 3),
        )
        """
        arguments = [self.motions, self.ground_truth, self.sequences, self.actions, self.indexes, self.skeleton]
        if shape == "dict":
            result = ac.accumulate_as_dict(*arguments, use_cache=self._use_cache)
        elif shape == "flat":
            result = ac.accumulate_as_tensor(*arguments, use_cache=self._use_cache)
        else:
            raise ValueError(f"Invalid shape: {shape}, must be 'dict' or 'flat'")

        self._use_cache = True
        return result

    def get_skeleton_names(self):
        skeleton_idxes = ut.dim_zero_cat(self.skeleton_idxes)
        assert (skeleton_idxes[0] == skeleton_idxes).all()
        return [KEYPOINT_NAMES[i] for i in skeleton_idxes[0]]

    def save_state(self, folder: Path, suffix: str = ""):
        motions, ground_truth = self.accumulate(shape="dict")

        big_dict = {}
        for sequence, data in motions.items():
            for action, motion in data["actions"].items():
                gt = ground_truth[sequence]["actions"][action]
                skel = data["skeleton"]

                key = f"{sequence}/{action}"
                big_dict[key] = {"motion": motion.half(), "skeleton": skel.short(), "ground_truth": gt.half()}

        # Save the big dictionary
        folder.mkdir(parents=True, exist_ok=True)
        save_file = folder / f"motion_data_{suffix}.pt"
        torch.save(big_dict, save_file, pickle_protocol=4)

        return save_file

    def get_world_prediction(self, prediction: dict, batch: dict):
        if "joints_3D" in prediction:
            joints3Dwr = prediction["joints_3D"].detach().cpu()  # Shape: (B, T, J, 3)
        elif "joints_3D_cc" in prediction:
            # We only evaluate the predictions from the left camera
            joints3Dcc = prediction["joints_3D_cc"].detach().cpu().float()
            cam_poses = batch["cam_poses"]["vr"].detach().cpu().float()
            joints3Dwr = geo.rototranslate(joints3Dcc, cam_poses)[:, :, 0]  # Shape: (B, T, J, 3)
        else:
            raise ValueError(f"No valid key found in prediction: {prediction.keys()}")

        return joints3Dwr


def encode_strings(strings: list[str], max_len: int = 30) -> Tensor:
    # Convert strings to list of ASCII values
    ascii_values = [[ord(c) for c in s] for s in strings]

    # Pad the sequences with zeros
    padded_ascii_values = [lst + [0] * (max_len - len(lst)) for lst in ascii_values]

    # Convert list of padded ASCII values to a tensor
    tensor = torch.tensor(padded_ascii_values, dtype=torch.uint8)
    return tensor
