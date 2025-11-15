import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import torch

from framevision.metrics.base import MotionMetric


class DetailedError(MotionMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = "mpjpe"

    def compute(self):
        global_error = self._compute_global()
        per_joint_errors = self._compute_per_joint()
        per_sequence_errors = self._compute_per_sequence()
        per_action_errors = self._compute_per_action()
        return global_error | per_joint_errors | per_sequence_errors | per_action_errors

    def plot(self):
        per_joint_fig = self._plot_per_joint()
        per_sequence_fig = self._plot_per_sequence()
        per_action_fig = self._plot_per_action()
        return per_joint_fig | per_sequence_fig | per_action_fig

    def _compute_global(self):
        motions, ground_truth = self.accumulate(shape="flat")  # Shape: (N, J, 3)
        aligned_motions = procrustes_align(motions, ground_truth)

        error = torch.norm(motions - ground_truth, dim=-1)
        aligned_error = torch.norm(aligned_motions - ground_truth, dim=-1)

        mpjpe = error.mean() * 1000  # Convert to mm
        pa_mpjpe = aligned_error.mean() * 1000  # Convert to mm
        return {f"{self.prefix}": mpjpe.item(), "pa-mpjpe": pa_mpjpe.item()}

    def _compute_per_joint(self):
        motions, ground_truth = self.accumulate(shape="flat")
        error = torch.norm(motions - ground_truth, dim=-1)  # Shape: (N, J)
        mpjpe = error.mean(dim=0) * 1000  # Shape: (J,)

        skeleton_names = self.get_skeleton_names()  # Shape: (J,)
        per_joint_error = {}
        for name, error in zip(skeleton_names, mpjpe):
            name = name.lower()
            if name.startswith("left"):
                name = name.replace("left", "")
            elif name.startswith("right"):
                name = name.replace("right", "")

            if name not in per_joint_error:
                per_joint_error[f"{self.prefix}/{name}"] = error.item()
            else:
                per_joint_error[f"{self.prefix}/{name}"] += error.item()
                per_joint_error[f"{self.prefix}/{name}"] /= 2  # Average left and right

        return per_joint_error

    def _compute_per_sequence(self):
        motions, ground_truth = self.accumulate(shape="dict")

        errors = {}
        for sequence, seq_data in motions.items():
            sequence_error = 0
            total_frames = 0
            for action in seq_data["actions"]:
                motion = seq_data["actions"][action]
                gt = ground_truth[sequence]["actions"][action]

                error = torch.norm(motion - gt, dim=-1)
                mpjpe = error.mean() * 1000  # Convert to mm

                num_frames = motion.shape[0]
                sequence_error += mpjpe.item() * num_frames
                total_frames += num_frames

            errors[sequence] = sequence_error / total_frames

        return {f"{self.prefix}/{sequence}": error for sequence, error in errors.items()}

    def _compute_per_action(self):
        motions, ground_truth = self.accumulate(shape="dict")

        errors = {}
        for sequence, seq_data in motions.items():
            for action in seq_data["actions"]:
                motion = seq_data["actions"][action]
                gt = ground_truth[sequence]["actions"][action]

                error = torch.norm(motion - gt, dim=-1)
                mpjpe = error.mean() * 1000  # Convert to mm

                if action not in errors:
                    errors[action] = {"total_error": 0, "total_frames": 0}

                num_frames = motion.shape[0]
                errors[action]["total_error"] += mpjpe.item() * num_frames
                errors[action]["total_frames"] += num_frames

        # Calculate average error per action
        for action in errors:
            errors[action] = errors[action]["total_error"] / errors[action]["total_frames"]

        return {f"{self.prefix}/{action}": error for action, error in errors.items()}

    def _plot_per_joint(self):
        # Prepare the data for plotting
        motions, ground_truth = self.accumulate(shape="flat")
        num_joints = motions.shape[-2]

        error_data = torch.norm(motions - ground_truth, dim=-1)  # Shape: (N, J)
        error_data_np = error_data.cpu().numpy() * 1000  # Convert to mm as in the original plot
        error_data_np = error_data_np.astype(np.float32)

        # Merge left and right joints
        merged_errors = {}
        for i, name in enumerate(self.get_skeleton_names()):
            name = name.lower()
            if name.startswith("left"):
                name = name.replace("left", "")
            elif name.startswith("right"):
                name = name.replace("right", "")

            if name not in merged_errors:
                merged_errors[name] = error_data_np[:, i]
            else:
                merged_errors[name] = np.concatenate([merged_errors[name], error_data_np[:, i]], axis=0)

        # Convert skeleton names to numpy array
        skeleton_names = np.array(self.get_skeleton_names())

        # Get a list of unique colors for each joint using Plotly's default color scale
        colors = pc.qualitative.Bold  # This is a built-in color palette with distinct colors

        # Create figure
        fig = go.Figure()

        # Compute stats and add boxplots in a single loop
        for i, (name, joint_errors) in enumerate(merged_errors.items()):
            # Precompute necessary statistics for each joint
            q1 = np.percentile(joint_errors, 25)  # 1st quartile (Q1)
            median = np.percentile(joint_errors, 50)  # Median (50th percentile)
            q3 = np.percentile(joint_errors, 75)  # 3rd quartile (Q3)
            iqr = q3 - q1  # Interquartile range

            lower_values = joint_errors[joint_errors >= (q1 - 1.5 * iqr)]
            upper_values = joint_errors[joint_errors <= (q3 + 1.5 * iqr)]

            if lower_values.size == 0 or upper_values.size == 0:
                return {}

            lower_whisker = np.min(lower_values)
            upper_whisker = np.max(upper_values)
            mean = np.mean(joint_errors)  # Mean

            # Add trace for each joint
            fig.add_trace(
                go.Box(
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower_whisker],
                    upperfence=[upper_whisker],
                    mean=[mean],
                    name=name,
                    x=[name],
                    width=0.5,
                    fillcolor=colors[i % len(colors)],  # Cycle through colors
                    line=dict(color="rgb(8,48,107)"),
                )
            )

        # Customize the layout
        skeleton_names = list(merged_errors.keys())
        fig.update_layout(
            title="Per Joint Pose Error",
            yaxis_title="Error (mm)",
            xaxis_title="Body Joints",
            xaxis=dict(tickmode="array", tickvals=np.arange(num_joints), ticktext=skeleton_names),
            template="plotly_white",
            boxmode="group",
            showlegend=False,
        )

        return {f"{self.prefix}/boxplot_per_joint": fig}

    def _plot_per_sequence(self):
        # Prepare the data for plotting
        motions, ground_truth = self.accumulate(shape="dict")

        sequence_errors = {}

        for sequence, seq_data in motions.items():
            errors = []
            for action in seq_data["actions"]:
                motion = seq_data["actions"][action]
                gt = ground_truth[sequence]["actions"][action]

                error = torch.norm(motion - gt, dim=-1)
                errors.extend(error.cpu().numpy() * 1000)  # Convert to mm and accumulate

            sequence_errors[sequence] = np.concatenate(errors, axis=0).astype(np.float32)

        # Create figure
        fig = go.Figure()

        # Get a list of unique colors for each sequence using Plotly's default color scale
        colors = pc.qualitative.Bold

        # Add trace for each sequence
        for i, (sequence, errors) in enumerate(sequence_errors.items()):
            # Compute statistics
            q1 = np.percentile(errors, 25)
            median = np.percentile(errors, 50)
            q3 = np.percentile(errors, 75)
            iqr = q3 - q1
            lower_values = np.array(errors)[np.array(errors) >= (q1 - 1.5 * iqr)]
            upper_values = np.array(errors)[np.array(errors) <= (q3 + 1.5 * iqr)]

            if lower_values.size == 0 or upper_values.size == 0:
                return {}

            lower_whisker = np.min(lower_values)
            upper_whisker = np.max(upper_values)
            mean = np.mean(errors)

            fig.add_trace(
                go.Box(
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower_whisker],
                    upperfence=[upper_whisker],
                    mean=[mean],
                    name=sequence,
                    x=[sequence],
                    width=0.5,
                    fillcolor=colors[i % len(colors)],
                    line=dict(color="rgb(8,48,107)"),
                )
            )

        # Customize the layout
        sequence_names = list(sequence_errors.keys())
        fig.update_layout(
            title="Per Sequence Pose Error",
            yaxis_title="Error (mm)",
            xaxis_title="Sequences",
            xaxis=dict(tickmode="array", tickvals=np.arange(len(sequence_names)), ticktext=sequence_names),
            template="plotly_white",
            boxmode="group",
            showlegend=False,
        )

        return {f"{self.prefix}/boxplot_per_sequence": fig}

    def _plot_per_action(self):
        # Prepare the data for plotting
        motions, ground_truth = self.accumulate(shape="dict")

        action_errors = {}

        for sequence, seq_data in motions.items():
            for action in seq_data["actions"]:
                motion = seq_data["actions"][action]
                gt = ground_truth[sequence]["actions"][action]

                error = torch.norm(motion - gt, dim=-1)
                if action not in action_errors:
                    action_errors[action] = []
                action_errors[action].extend(error.cpu().numpy().astype(np.float32) * 1000)  # Convert to mm

        # Create figure
        fig = go.Figure()

        # Get a list of unique colors for each action using Plotly's default color scale
        colors = pc.qualitative.Bold

        # Add trace for each action
        for i, (action, errors) in enumerate(action_errors.items()):
            # Compute statistics
            q1 = np.percentile(errors, 25)
            median = np.percentile(errors, 50)
            q3 = np.percentile(errors, 75)
            iqr = q3 - q1

            lower_values = np.array(errors)[np.array(errors) >= (q1 - 1.5 * iqr)]
            upper_values = np.array(errors)[np.array(errors) <= (q3 + 1.5 * iqr)]

            if lower_values.size == 0 or upper_values.size == 0:
                return {}

            lower_whisker = np.min(lower_values)
            upper_whisker = np.max(upper_values)
            mean = np.mean(errors)

            fig.add_trace(
                go.Box(
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower_whisker],
                    upperfence=[upper_whisker],
                    mean=[mean],
                    name=action,
                    x=[action],
                    width=0.5,
                    fillcolor=colors[i % len(colors)],
                    line=dict(color="rgb (8,48, 107) "),
                )
            )

        #  Customize the layout
        action_names = list(action_errors.keys())
        fig.update_layout(
            title="Per Action  Pose Error",
            yaxis_title="Error (mm)",
            xaxis_title="Actions",
            xaxis=dict(tickmode="array", tickvals=np.arange(len(action_names)), ticktext=action_names),
            template="plotly_white",
            boxmode="group",
            showlegend=False,
        )

        return {f"{self.prefix}/boxplot_per_action": fig}


def procrustes_align(X, Y):
    """Aligns each pose in X to the corresponding pose in Y independently using Procrustes analysis."""
    original_dtype = X.dtype
    X, Y = X.to(torch.float32), Y.to(torch.float32)

    # Center the poses by subtracting the mean for each pose independently
    X_mean = X.mean(dim=1, keepdim=True)  # (N, 1, 3)
    Y_mean = Y.mean(dim=1, keepdim=True)  # (N, 1, 3)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Compute the optimal rotation matrix using SVD
    covariance_matrices = Y_centered.transpose(1, 2) @ X_centered  # (N, 3, 3)
    try:
        U, _, Vt = torch.linalg.svd(covariance_matrices)  # U, Vt: (N, 3, 3)
        R = U @ Vt  # (N, 3, 3)
    except RuntimeError:
        # If SVD fails, return zeros of the proper shape
        R = torch.zeros_like(covariance_matrices)

    # Apply the rotation to X_centered
    X_aligned = X_centered @ R  # (N, J, 3)

    # Return the aligned poses by adding the mean of the ground truth poses back
    X = X_aligned + Y_mean

    return X.to(original_dtype)
