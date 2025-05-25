import torch


class OnlineStats:
    """Online statistics for tensors with fixed first dim (3) and variable second dim."""

    def __init__(self, num_features=16):
        self.num_features = num_features
        self.count = 0  # Total number of values per feature
        self.mean = torch.zeros(num_features, dtype=torch.float32)  # Per-feature mean
        self.sum_squared_diff = torch.zeros(num_features, dtype=torch.float32)  # Per-feature sum of squared differences

    def update(self, x: torch.Tensor):
        """Update stats with tensor x of shape [16, length], where length can vary."""
        assert x.shape[0] == self.num_features, f"First dimension must be {self.num_features}, got {x.shape[0]}"
        batch_count = x.shape[1]  # Length of current sample
        self.count += batch_count

        # Compute mean and sum of squared differences for each feature
        batch_mean = x.mean(dim=1)  # Shape: [3]
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / self.count  # Update running mean

        # Update sum of squared differences (Welford's algorithm)
        for i in range(x.shape[1]):  # Iterate over length
            delta_x = x[:, i] - self.mean
            self.sum_squared_diff += delta_x * (x[:, i] - batch_mean)

    def get_mean(self):
        """Return per-feature mean or None if no samples."""
        return self.mean if self.count > 0 else None

    def get_std(self):
        """Return per-feature standard deviation or None if insufficient samples."""
        if self.count < 2:
            return None
        variance = self.sum_squared_diff / (self.count - 1)  # Bessel's correction
        return torch.sqrt(variance + 1e-8)  # Std with stability constant
