"""
Text-based parser for ProteinNet Records.
Disclaimer: 
- Part of the code is credited to https://github.com/aqlaboratory/proteinnet/blob/master/code/tf_parser.py
"""
import re
import numpy as np
import h5py

# Constants
NUM_DIMENSIONS = 3

# Functions for conversion from Mathematica protein files to TFRecords
_aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
_dssp_dict = {'L': '0', 'H': '1', 'B': '2', 'E': '3', 'G': '4', 'I': '5', 'T': '6', 'S': '7'}
_mask_dict = {'-': '0', '+': '1'}


class switch(object):
  """Switch statement for Python, based on recipe from Python Cookbook."""

  def __init__(self, value):
    self.value = value
    self.fall = False

  def __call__(self, *args):
    """Indicate whether or not to enter a case suite."""
    if self.fall or not args:
      return True
    elif self.value in args: # changed for v1.5
      self.fall = True
      return True
    else:
      return False


def letter_to_num(string, dict_):
  """Convert string of letters to list of ints."""
  patt = re.compile('[' + ''.join(dict_.keys()) + ']')
  num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
  num = [int(i) for i in num_string.split()]
  return num


def read_record(file_, num_evo_entries):
  """ Read a Mathematica protein record from file and convert into dict. """
  dict_ = {}

  while True:
    next_line = file_.readline()
    case = switch(next_line)
    if case('[ID]' + '\n'):
      id_ = file_.readline()[:-1]
      dict_.update({'id': id_})
    elif case('[PRIMARY]' + '\n'):
      primary = letter_to_num(file_.readline()[:-1], _aa_dict)
      dict_.update({'primary': primary})
    elif case('[EVOLUTIONARY]' + '\n'):
      evolutionary = []
      for residue in range(num_evo_entries):
          evolutionary.append([float(step) for step in file_.readline().split()])
      dict_.update({'evolutionary': evolutionary})
    elif case('[SECONDARY]' + '\n'):
      secondary = letter_to_num(file_.readline()[:-1], _dssp_dict)
      dict_.update({'secondary': secondary})
    elif case('[TERTIARY]' + '\n'):
      tertiary = []
      for axis in range(NUM_DIMENSIONS): 
        tertiary.append([float(coord) for coord in file_.readline().split()])
      dict_.update({'tertiary': tertiary})
    elif case('[MASK]' + '\n'):
      mask = letter_to_num(file_.readline()[:-1], _mask_dict)
      dict_.update({'mask': mask})
    elif case('\n'):
      return dict_
    elif case(''):
      return None


def text_to_hdf(infile, outfile, num_evo_entry=21):
  """Data re-formatting from text-based to hdf."""

  int32_t = h5py.vlen_dtype(np.dtype('int32'))
  # TODO: use float64?

  float32_t = h5py.vlen_dtype(np.dtype('float32'))
  str_t = h5py.string_dtype(encoding='utf-8')

  with h5py.File(outfile, 'w') as f:
    id_dset = f.create_dataset('id', shape=(0,), maxshape=(None,), dtype=str_t)
    pri_dset = f.create_dataset('primary', shape=(0,), maxshape=(None,), dtype=int32_t)
    evo_dset = f.create_dataset('evolutionary', shape=(0, 21,), maxshape=(None, 21,), dtype=float32_t)
    ter_dset = f.create_dataset('tertiary', shape=(0, 3,), maxshape=(None, 3,), dtype=float32_t)
    msk_dset = f.create_dataset('mask', shape=(0,), maxshape=(None,), dtype=int32_t)

    idx = 0
    in_obj = open(infile, 'r')

    while True:
      data_dict = read_record(in_obj, num_evo_entry)
      if data_dict is None:
        return

      id_dset.resize(idx + 1, axis=0)
      pri_dset.resize(idx + 1, axis=0)
      evo_dset.resize(idx + 1, axis=0)
      ter_dset.resize(idx + 1, axis=0)
      msk_dset.resize(idx + 1, axis=0)

      id_dset[idx] = data_dict['id']
      pri_dset[idx] = data_dict['primary']
      evo_dset[idx] = data_dict['evolutionary']
      ter_dset[idx] = data_dict['tertiary']
      msk_dset[idx] = data_dict['mask']

      idx += 1



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Convert text-based Proteinnet to HDF5 dataset.')
  parser.add_argument('--inpath', metavar='fname', type=str, help='path to input data')
  #parser.add_argument('--outdir', metavar='fname', type=str, help='dir to save output HDF5 dataset')
  parser.add_argument('--num_evo_entry', metavar='N', type=int, default=21, help='number of evolutionary entries')
  args = parser.parse_args()

  outpath = args.inpath + '.hdf5'

  text_to_hdf(args.inpath, outpath, args.num_evo_entry)
