import torch


class FinsConditionalBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features: int, condition_length: int):
        super().__init__()

        self.num_features = num_features
        self.condition_length = condition_length
        self.norm = torch.nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)

        self.layer = torch.nn.utils.spectral_norm(torch.nn.Linear(condition_length, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)
        self.layer.bias.data.zero_()

    def forward(self, inputs: torch.Tensor, noise: torch.Tensor):
        outputs: torch.Tensor = self.norm(inputs)
        gamma: torch.Tensor
        beta: torch.Tensor
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)

        outputs = gamma * outputs + beta

        return outputs
