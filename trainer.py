import os
import time
import torch
from metric import compute_drmsd_over_batch


class Trainer():
  def __init__(self, model, train_loader, val_loader, config):

    self.cfg = config['train']
    self.NUM_DIHEDRALS = config['model']['num_dihedrals']
    self.model = model
    self.dataloader = {'train': train_loader,
                       'val': val_loader}

    self._cuda()

    self._start_epoch = 0
    self._global_step = 0

    self.optimizer = torch.optim.Adam(self.model.parameters())

  def _cuda(self):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(self.device)

  def _net_mode(self, is_train):
    if is_train:
      self.model.train()
    else:
      self.model.eval()

  def loss_fn(self, x, y, m, s):
    if self.cfg['atoms'] == 'c_alpha':
      x = x[1::self.NUM_DIHEDRALS]
      y = y[1::self.NUM_DIHEDRALS]
    return compute_drmsd_over_batch(x, y, m, s)

  def train(self):
    for epoch in range(self._start_epoch, self.cfg['epochs']):
      self._net_mode(is_train=True)
      epoch_loss = 0
      start = time.time()
      for step, inputs in enumerate(self.dataloader['train']):
        # check empty data stream when all samples exceed max_len
        if inputs is None:
          continue

        prim, evol, tert, mask, slen = [x.to(self.device) for x in inputs]
        coord = self.model([prim, evol, slen])
        loss, loss_sum = self.loss_fn(coord, tert, mask, slen)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss_sum.data

        self._global_step += 1

      avg_loss = epoch_loss / len(self.dataloader['train'].dataset)
      print("Epoch: {} | Avg drmsd: {:.3f} | Time: {:.2f}".format(epoch + 1,
                                                                  avg_loss,
                                                                  time.time() - start))

      if (epoch + 1) % self.cfg['save_step'] == 0:
        self.save(epoch)

      if (epoch + 1) % self.cfg['eval_step'] == 0:
        self.evaluate()

  def evaluate(self, dl=None):
    if dl is None:
      dl = self.dataloader['val']

    self._net_mode(is_train=False)
    eval_loss = 0
    start = time.time()
    for step, inputs in enumerate(dl):
      if inputs is None:
        continue
      prim, evol, tert, mask, slen = [x.to(self.device) for x in inputs]
      coord = self.model([prim, evol, slen])
      loss, loss_sum = self.loss_fn(coord, tert, mask, slen)
      eval_loss += loss_sum.data

    avg_loss = eval_loss / len(dl.dataset)
    print("Evaluate loss: {:.3f} | Time: {:.2f}".format(avg_loss, time.time() - start))


  def save(self, epoch):
    path = os.path.join(self.cfg['save_path'], "epoch_{}".format(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

  def load(self, fname):
    ckpt = torch.load(fname)
    self.model.load_state_dict(ckpt['model_state_dict'])
    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    self._start_epoch = ckpt['epoch']



class Logger():
  def __init__(self, log_dir):
    self.writer = torch.utils.tensorboard.SummaryWriter()

  def scalar_summary(self, tag, value, step):
    pass

  def image_summary(self,):
    pass

  def hist_summary(self,):
    pass


