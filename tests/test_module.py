import torch

from gmm_linear import GMMLinear


@torch.no_grad()
def test_sampling():
    n_features = 100
    layer = GMMLinear(n_features, 8, 3)
    layer.eval()

    data = torch.randn(1, n_features)
    indices = [torch.randint(0, data.shape[1], (1,))]

    incomplete = data.clone()
    incomplete[0, indices] = float('nan')
    incomplete = layer(incomplete)

    n = 20_000
    batch_size = 100
    sampled = torch.zeros_like(incomplete)
    for _ in range(n):
        sample = torch.randn(batch_size, *layer.mean.shape) * torch.exp(layer.log_std2) + layer.mean
        sample = (sample * torch.softmax(layer.log_weights, 0)[:, None]).sum(1)
        local = data.repeat_interleave(batch_size, 0)
        local[:, indices] = sample[:, indices]
        sampled += torch.relu(layer.linear(local)).mean(0, keepdim=True)

    sampled /= n
    torch.testing.assert_allclose(incomplete, sampled, atol=0.01, rtol=0.0001)
