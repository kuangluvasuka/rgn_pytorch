from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geom_ops import angularize, dihedral_to_point, point_to_coordinate


class RGNModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    cfg = deepcopy(config['model'])

    self.register_buffer('BOND_LENGTHS', torch.tensor([145.801, 152.326, 132.868], dtype=torch.float32))
    self.register_buffer('BOND_ANGLES', torch.tensor([2.124, 1.941, 2.028], dtype=torch.float32))
    self.register_buffer('INIT_MAT', torch.from_numpy(
        np.array([[-np.sqrt(1. / 2.), np.sqrt(3. / 2.), 0.],
                  [-np.sqrt(2.), 0., 0.],
                  [0., 0., 0.]], dtype=np.float32)))

    self.rnn_module = RecurrentLayer(config)

    dihedral_fc_layer_dim = deepcopy(cfg['dihedral_fc_layer_dim'])
    dihedral_fc_layer_dim.insert(0, self.rnn_module.get_output_size())

    self.dihedral = DihedralLayer(cfg['alphabet_size'], cfg['num_dihedrals'], dihedral_fc_layer_dim,
                                  cfg['use_alphabet'], cfg['non_linear_dihedral'],
                                  cfg['is_angularized'], cfg['alphabet_temperature'], act='tanh')

  def forward(self, inputs):
    """
    Note: inputs contain 3 elements -- primary, evolutionary, seq_length
    """
    recurrent_output = self.rnn_module(inputs)
    # recurrent_output size is [B, T, rnn_output_dim]
    dihedral_angles = self.dihedral(recurrent_output)
    points = dihedral_to_point(dihedral_angles, self.BOND_LENGTHS, self.BOND_ANGLES)
    coordinates = point_to_coordinate(points, self.INIT_MAT)

    return coordinates


class DihedralLayer(nn.Module):
  def __init__(self, alphabet_size, num_dihedrals, dihedral_dims,
               use_alphabet=True, non_linear_dihedral=True,
               is_angularized=True, alphabet_temperature=1., act='tanh'):

    super().__init__()

    activation = nn.ModuleDict({'relu': nn.ReLU(),
                                'tanh': nn.Tanh()})

    # save hparams for forward()
    self._use_alphabet = use_alphabet
    self._is_angularized = is_angularized
    self._alphabet_temperature = alphabet_temperature

    if self._use_alphabet:
      dihedral_dims.append(alphabet_size)
      param = torch.randn(alphabet_size, num_dihedrals)
      if self._is_angularized:
        param = angularize(torch.randn(alphabet_size, num_dihedrals))
      self.alphabet_weight = nn.Parameter(param)
    else:
      dihedral_dims.append(num_dihedrals)

    if non_linear_dihedral:
      fc = []
      for i in range(len(dihedral_dims) - 2):
        fc.append(nn.Linear(dihedral_dims[i], dihedral_dims[i + 1], bias=True))
        # TODO: nomalization?

        fc.append(activation[act])
      fc.append(nn.Linear(dihedral_dims[-2], dihedral_dims[-1], bias=True))
      self.dihedral_fc = nn.Sequential(*fc)
    else:
      self.dihedral_fc = nn.Linear(dihedral_dims[0], dihedral_dims[-1], bias=True)


  def _reduce_to_mean(self, weights):
    sins = torch.sin(self.alphabet_weight)
    coss = torch.cos(self.alphabet_weight)
    y_coords = torch.matmul(weights, sins)
    x_coords = torch.matmul(weights, coss)
    return torch.atan2(y_coords, x_coords)

  def forward(self, rnn_acts):
    """
    Inputs: rnn activations [T, B, rnn_out_dim]

    Return:
      dihedral angles [T, B, num_dihedrals=3]
    """
    linear = self.dihedral_fc(rnn_acts)
    # TODO: add nomalization?

    if self._use_alphabet:
      seq_len, b_size, alphabet_size = linear.shape
      flattened_linear = linear.view(-1, alphabet_size)
      probs = F.softmax(flattened_linear / self._alphabet_temperature, dim=1)
      flattened_dihedrals = self._reduce_to_mean(probs)
      dihedrals = flattened_dihedrals.view(seq_len, b_size, -1)
    else:
      dihedrals = linear
      if self._is_angularized:
        dihedrals = angularize(dihedrals)

    # TODO: add angle shift

    # [T, B, dihedral_dim=3]
    return dihedrals


class RecurrentLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    cfg = deepcopy(config['model'])
    self.skip_connection = cfg['rnn_skip_connection']
    self.bidirectional = cfg['bidirectional']
    self.layers = nn.ModuleList()

    # 0. ModuleDict, ParameterDict
    # 1. if higher_order_layers, if skip connections
    # 2. for each layer, if bidirectional

    # residue embedding
    if cfg['use_residue_embedding']:
      self.residue_embedding = nn.Embedding(cfg['num_residue'] + 1, cfg['embedding_dim'], padding_idx=0)
      init_size = cfg['embedding_dim'] + config['data']['num_evo_entry']
    else:
      one_hot = torch.eye(cfg.num_residue + 1)
      init_size = cfg['num_residue'] + 1 + config['data']['num_evo_entry']
      self.residue_embedding = nn.Embedding.from_pretrained(one_hot, freeze=True, padding_idx=0)

    cfg['rnn_hidden_size'] = deepcopy(cfg['rnn_layer_size'])
    cfg['rnn_layer_size'].pop()
    if cfg['bidirectional']:
      cfg['rnn_layer_size'] = [x * 2 for x in cfg['rnn_layer_size']]
    cfg['rnn_layer_size'].insert(0, init_size)

    # select rnn module
    if cfg['rnn_cell'].lower() == 'vanilla':
      rnn_class = nn.RNN
    elif cfg['rnn_cell'].lower() == 'lstm':
      rnn_class = nn.LSTM
    elif cfg['rnn_cell'].lower() == 'gru':
      rnn_class = nn.GRU
    else:
      raise ValueError("invalid option '%s' for recurrent layer." % cfg['rnn_cell'])

    # TODO: bidirectional
    for layer_idx, (input_dim, hidden_dim, num_stacks, dropout) in enumerate(
            zip(cfg['rnn_layer_size'], cfg['rnn_hidden_size'],
                cfg['rnn_layer_stacks'], cfg['rnn_layer_dropout'])):
      self.layers.append(rnn_class(input_dim, hidden_dim, num_stacks,
                                   dropout=dropout, bidirectional=cfg['bidirectional']))

  def get_output_size(self):
    bidirectional_multiplier = 2 if self.bidirectional else 1
    skip_addition = self.layers[0].input_size if self.skip_connection else 0
    return self.layers[-1].hidden_size * bidirectional_multiplier + skip_addition

  def _pack_inputs(self, inputs):
    """Reformatting inputs into torch.packed_sequence"""
    # primary=[T, B], evol=[T, B, 21]
    primary, evolutionary, seq_length = inputs

    embed = self.residue_embedding(primary)
    x = torch.cat((embed, evolutionary), dim=2)
    packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_length, enforce_sorted=False)
    return packed_x, x

  def forward(self, inputs):
    """inputs: [primary, evolutionary, seq_length]
    """
    packed_input, inputs_ = self._pack_inputs(inputs)

    layer_outputs = [packed_input]
    for i in range(len(self.layers)):
      # NOTE: use default zero-valued vector as initial hidden state
      layer_out, _ = self.layers[i](layer_outputs[-1])
      layer_outputs.append(layer_out)

    output, _ = nn.utils.rnn.pad_packed_sequence(layer_outputs[-1])

    if self.skip_connection:
      output = torch.cat((output, inputs_), 2)

    # output size is [T, B, output_dim]

    return output
