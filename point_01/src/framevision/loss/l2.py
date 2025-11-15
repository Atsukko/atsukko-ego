import torch
import torch.nn as nn
from torch import Tensor


class L2Loss(nn.Module):
    """Compute the L2 loss.

    Args:
        prediction_key: Key to access body_tracking data on the prediction. Usually one of:
            - "joints_3D", the global 3D joints.
            - "joints_3D_cc", the 3D joints in camera coordinate.
        target_key: Key to access body_tracking data on the batch. Usually the same as prediction_key.
        output_key: Key to store the loss in the output dictionary.
    """

    def __init__(
        self,
        prediction_key: str,
        target_key: str = "",
        output_key: str = "l2",
    ):
        super().__init__()
        self.prediction_key = prediction_key
        self.target_key = target_key or prediction_key
        self.output_key = output_key

    def forward(self, prediction: dict, batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute Mean Per Joint Position Error (loss).

        Args:
            batch: The batch of data, must contain -> "body_tracking" -> key.
                predicted_joints: Predicted 3D joints. Shape: (..., J, 3)
            prediction: The prediction from the model, must contain -> key.
                joints_3d: Ground truth 3D joints. Shape: (..., J, 3) OR list of tensors of shapes (..., J, 3)

        Returns:
            loss: The loss value.
            output: A dictionary with the non-reduced losses. Necessary for logging.
        """
        predicted_joints, gt_joints = prediction[self.prediction_key], batch["body_tracking"][self.target_key]

        # If the prediction is a tensor, compute the loss
        if isinstance(predicted_joints, Tensor):
            loss = self._compute_error(predicted_joints, gt_joints)
            return loss.mean(), {self.output_key: loss}

        # If the prediction is a list of tensors, compute the loss for each tensor and average them
        loss = sum(self._compute_error(pred, gt_joints) for pred in predicted_joints) / len(predicted_joints)
        return loss.mean(), {self.output_key: loss}

    def _compute_error(self, predicted_joints: Tensor, gt_joints: Tensor):
        error = torch.norm(predicted_joints - gt_joints, dim=-1)  # L2 norm
        # Average all dimensions but the first
        return error.mean(dim=tuple(range(1, error.dim())))
