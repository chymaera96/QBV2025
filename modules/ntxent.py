import torch
import torch.nn.functional as F


def ntxent_loss(z_i, z_j, cfg):
    tau = cfg['tau']
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)

    sim = torch.matmul(z, z.T) / tau

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    positives = sim[mask].view(2 * batch_size, 1)

    negatives = sim[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(2 * batch_size, device=z.device, dtype=torch.long)

    loss = F.cross_entropy(logits, labels)
    return loss