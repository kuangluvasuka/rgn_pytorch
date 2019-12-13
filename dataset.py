"""Dataset class defined here"""

import h5py
import numpy as np
import torch


class ProteinDataset(torch.utils.data.Dataset):
  """Dataset class that Loads data from hdf5 collections."""
  def __init__(self, filename, config):
    super().__init__()
    self.max_sequence_length = config['data']['max_sequence_length']
    self.f = h5py.File(filename, 'r')

  def __getitem__(self, idx):
    id_str = self.f['id'][idx]
    # NOTE: add 1 to make 0-value as padding value
    prim = self.f['primary'][idx] + 1

    # discard sequences longer than max
    seq_length = len(prim)
    if(seq_length > self.max_sequence_length):
      return None
    evol = np.stack(self.f['evolutionary'][idx], axis=0)
    tert = np.stack(self.f['tertiary'][idx], axis=0)
    mask = self.f['mask'][idx].astype(np.float32)
    return torch.tensor(prim, dtype=torch.int64), torch.tensor(evol, dtype=torch.float32), \
           torch.tensor(tert, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), \
           seq_length, id_str

  def __len__(self):
    return len(self.f['id'])

  @staticmethod
  def collate_fn(samples):
    """Mini-batch"""

    def pad_tensor(v, out_length, dim):
      pad_shape = list(v.shape)
      pad_shape[dim] = out_length - v.shape[dim]
      return torch.cat([v, torch.zeros(*pad_shape, dtype=v.dtype)], dim=dim)

    samples = list(filter(lambda x: x is not None, samples))
    if (len(samples) == 0):
      return None

    seq_lengths = torch.tensor([x[4] for x in samples])
    max_len = torch.max(seq_lengths)

    prim_batch_major = torch.stack(list(map(lambda x: pad_tensor(x[0], max_len, 0), samples)), dim=0)
    evol_batch_major = torch.stack(list(map(lambda x: pad_tensor(x[1], max_len, 1), samples)), dim=0)
    tert_batch_major = torch.stack(list(map(lambda x: pad_tensor(x[2], max_len * 3, 1), samples)), dim=0)
    mask_batch_major = torch.stack(list(map(lambda x: pad_tensor(x[3], max_len, 0), samples)), dim=0)
    # batch_prim  & batch_mask [B, max_len],
    # batch_evol [B, 21, max_len], batch_tert [B, 3, 3*max_len]

    # NOTE: convert to time-major
    # primaries=[max_len, B], masks=[max_len, B], evol=[max_len, B, 21]
    primaries = prim_batch_major.permute(1, 0)
    evolutionaries = evol_batch_major.permute(2, 0, 1)
    tertiaries = tert_batch_major.permute(2, 0, 1)
    masks = mask_batch_major.permute(1, 0)

    return primaries, evolutionaries, tertiaries, masks, seq_lengths


    #TODO: pin memory!!!!



def get_dataloader(batch_size, dataset):
  return torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False, sampler=None,
      batch_sampler=None, num_workers=0, collate_fn=ProteinDataset.collate_fn,
      pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
