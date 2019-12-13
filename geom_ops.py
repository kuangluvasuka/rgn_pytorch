from collections import namedtuple
from math import pi as PI
import numpy as np
import torch
from torch.nn import functional as F


def angularize(v):
  """ Restricts real-valued tensors to the interval [-pi, pi] by feeding them through a cosine. """
  return torch.mul(PI, torch.cos(v + (PI / 2)))


def dihedral_to_point(dihedrals, bond_lengths, bond_angles):
  seq_len, b_size, dihedral_dim = dihedrals.shape

  r_cos_theta = bond_lengths * torch.cos(PI - bond_angles)
  r_sin_theta = bond_lengths * torch.sin(PI - bond_angles)

  point_x = r_cos_theta.view(1, 1, -1).repeat(seq_len, b_size, 1)
  point_y = torch.mul(torch.cos(dihedrals), r_sin_theta)
  point_z = torch.mul(torch.sin(dihedrals), r_sin_theta)

  point = torch.stack([point_x, point_y, point_z])
  point_perm = point.permute(1, 3, 2, 0)
  point_final = torch.reshape(point_perm, (seq_len * dihedral_dim, b_size, -1))

  return point_final


def point_to_coordinate(points, init_mat, num_fragments=6):
  # compute optimal number of fragments if needed
  total_num_angles, b_size, num_dimens = points.shape
  if num_fragments is None:
    num_fragments = int(np.sqrt(total_num_angles))

  # initial three coordinates (specifically chosen to eliminate need for extraneous matmul)
  Triplet = namedtuple('Triplet', 'a, b, c')
  init_coords = [row.repeat([num_fragments * b_size, 1]).view(num_fragments, b_size, num_dimens) for row in init_mat]
  init_coords = Triplet(*init_coords)

  padding = ((num_fragments - (total_num_angles % num_fragments)) % num_fragments)
  points = F.pad(points, (0, 0, 0, 0, 0, padding))
  points = points.view(num_fragments, -1, b_size, num_dimens).permute(1, 0, 2, 3)   # [FRAG_SIZE, NUM_FRAGS,  BATCH_SIZE, NUM_DIMENSIONS]

  def extend(tri, point, multi_m):
    bc = F.normalize(tri.c - tri.b, dim=-1)
    n = F.normalize(torch.cross(tri.b - tri.a, bc), dim=-1)
    if multi_m:
      m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 3, 0)
    else:
      s = point.shape + (3, )
      m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 0)
      m = m.repeat(s[0], 1, 1).view(s)
    coord = torch.squeeze(torch.matmul(m, point.unsqueeze(3)), dim=3) + tri.c
    return coord
  # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batche
  coords_list = [None] * points.shape[0]
  tri = init_coords

  for i in range(points.shape[0]):
    coord = extend(tri, points[i], True)
    coords_list[i] = coord
    tri = Triplet(tri.b, tri.c, coord)

  coords_pretrans = torch.stack(coords_list).permute(1, 0, 2, 3)

  # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
  coords_trans = coords_pretrans[-1]
  for i in reversed(range(coords_pretrans.shape[0] - 1)):
    transformed_coords = extend(Triplet(*[di[i] for di in tri]), coords_trans, False)
    coords_trans = torch.cat([coords_pretrans[i], transformed_coords], 0)

  coords = F.pad(coords_trans[: total_num_angles - 1], (0, 0, 0, 0, 1, 0))

  return coords
