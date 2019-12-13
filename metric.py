import torch


def pairwise_distance(u):
  """ Computes the pairwise distance (l2-norm) between all pairs of nodes in a sequence.
      Vectors are assumed to be in the third dimension.

  Args:
    u: [T, B, 3]
  Returns:
    [T, T, B]
  """
  diffs = u - u.unsqueeze(1)
  norms = torch.norm(diffs, dim=3)
  return norms


def compute_drmsd_over_batch(preds, targets, masks, seq_len):
  """ Computes reduced dRMSD loss between predicted tertiary structures and targets.

  Returns:
    drmsd_mean: scalar value for computing gradients
    drmsd_sum: scalar value for logging
  """
  diffs = pairwise_distance(preds) - pairwise_distance(targets)  # [T, T, B]
  diffs_masked = torch.mul(diffs, masks)
  norms = torch.norm(diffs_masked, dim=(0, 1))
  drmsd_ = 2 * torch.div(norms, torch.mul(seq_len, (seq_len - 1)))
  drmsd_mean = drmsd_.mean()
  drmsd_sum = drmsd_.sum()
  return drmsd_mean, drmsd_sum
