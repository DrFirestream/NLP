import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxml import etree

from tqdm import tqdm
from pytorch_extras import RAdam, SingleCycleScheduler
from pytorch_transformers import GPT2Model, GPT2Tokenizer
from deps.torch_train_test_loop.torch_train_test_loop import LoopComponent, TrainTestLoop

from models import SSTClassifier
from lm.inference import ModelWrapper
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
DEVICE = 'cuda'
model_path = '/Home/ode/pimenov/Documents/sirprey/Streets/nlp/de345-root'
mw = ModelWrapper.load(Path(model_path))
MAX_LEN = 0#395 #0 for routing
TOT_MAX_LEN = 512 
_stoi = { s: i for i, s in enumerate(['negative', 'neutral', 'positive'])}
create_batch = lambda :  type('', (), {'text':([], []), 'label':[], '__len__': lambda self: len(self.label)})()
def load_xml(fname, pad_token='', resample = None):
    if not pad_token:
        pad_token = mw.END_OF_TEXT
    r = etree.parse(fname).getroot()
    clssizes = [0 for i in range(len(_stoi))]
    texts = [([*mw.tokenize(d.getchildren()[-1].text)[:(min(MAX_LEN, TOT_MAX_LEN) if MAX_LEN else TOT_MAX_LEN) - 1], pad_token], _stoi[d.getchildren()[-2].text]) for d in r.getchildren()]
    print('Too long texts:%d out of %d'%( sum(1 for tokens,_ in texts if len(tokens)>1024), len(texts)))
    maxlen = MAX_LEN if MAX_LEN else max(len(tokens) for tokens,_ in texts)
    print('Maximal length of text:%d'%(maxlen))
    inputs = [([mw.token_to_id(token) for token in tokens + [pad_token] * (maxlen - len(tokens))], [1.0] * len(tokens) + [0.0] * (maxlen - len(tokens)), label) for tokens, label in texts]
    outputs = []
    for tokens, mask, label in inputs:
        if resample:
            for i in range(resample[label]):
                outputs.append((tokens, mask, label))
        clssizes[label] += 1
    print('Class sizes:' + str(clssizes))
    return outputs if resample else inputs
#load_xml('test_TIMESTAMP1.xml')
class LoopMain(LoopComponent):

    def __init__(self, n_classes, device, pct_warmup=0.1, mixup=(0.2, 0.2)):
        self.n_classes, self.device, self.pct_warmup = (n_classes, device, pct_warmup)
        self.mixup_dist = torch.distributions.Beta(torch.tensor(mixup[0]), torch.tensor(mixup[1]))
        self.onehot = torch.eye(self.n_classes, device=self.device)
        self.saved_data = []

    def on_train_begin(self, loop):
        n_iters = len(loop.train_data) * loop.n_epochs
        loop.optimizer = RAdam(loop.model.parameters(), lr=1e-3)#5e-4)
        loop.scheduler = SingleCycleScheduler(
            loop.optimizer, loop.n_optim_steps, frac=self.pct_warmup, min_lr=1e-5)
        
    def on_grads_reset(self, loop):
        loop.model.zero_grad()
    def on_batch_begin(self, loop):
        if loop.is_training:
            for i in range(len(_stoi)):
                loop.tp[i] //= 2
                loop.fp[i] //= 2
                loop.fn[i] //= 2
    def on_epoch_begin(self, loop):
        loop.tp = [0] * len(_stoi)
        loop.fp = [0] * len(_stoi)
        loop.fn = [0] * len(_stoi)
        #loop.precision = 0
        #loop.recall = 0
        #loop.f1 = 0
        loop.p1s = [0] * len(_stoi)
        loop.r1s = [0] * len(_stoi)
        loop.f1s = [0] * len(_stoi)
    def on_forward_pass(self, loop):
        model, batch = (loop.model, loop.batch)
        mask, embs = batch.text
        target_probs = self.onehot[batch.label]
        if loop.is_training:
            r = self.mixup_dist.sample([len(mask)]).to(device=mask.device)
            idx = torch.randperm(len(mask))
            mask = mask.lerp(mask[idx], r[:, None])
            embs = embs.lerp(embs[idx], r[:, None, None, None])
            target_probs = target_probs.lerp(target_probs[idx], r[:, None])

        pred_scores, _, _ = model(mask, embs)
        _, pred_ids = pred_scores.max(-1)
        for i in range(len(_stoi)):
            loop.tp[i] += torch.sum((pred_ids == i)*(batch.label == i))
            loop.fp[i] += torch.sum((pred_ids == i)*(batch.label != i))
            loop.fn[i] += torch.sum((pred_ids != i)*(batch.label == i))
        #loop.precision = [loop.tp.float().item() /(loop.fp.item()+loop.tp.item()+1e-3) 
        #loop.recall = loop.tp.float().item()/(loop.tp.item() + loop.fn.item()+1e-3)
        #loop.f1 = 2 * loop.precision * loop.recall / (loop.precision + loop.recall + 1e-3)
        loop.p1s = [loop.tp[i].float().item() /(loop.fp[i].item()+loop.tp[i].item()+1e-3) for i in range(len(_stoi))]
        loop.r1s = [loop.tp[i].float().item()/(loop.tp[i].item() + loop.fn[i].item()+1e-3) for i in range(len(_stoi))]
        loop.f1s = [2 * loop.p1s[i] * loop.r1s[i] / (loop.p1s[i] + loop.r1s[i] + 1e-3) for i in range(len(_stoi))]
        accuracy = (pred_ids == batch.label).float().mean()

        loop.pred_scores, loop.target_probs, loop.accuracy = (pred_scores, target_probs, accuracy)
    def on_loss_compute(self, loop):
        losses = -loop.target_probs * F.log_softmax(loop.pred_scores, dim=-1)  # CE
        loop.loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch

    def on_backward_pass(self, loop):
        loop.loss.backward()

    def on_optim_step(self, loop):
        loop.optimizer.step()
        loop.scheduler.step()

    def on_batch_end(self, loop):
        self.saved_data.append({
            'n_samples': len(loop.batch),
            'epoch_desc': loop.epoch_desc,
            'epoch_num': loop.epoch_num,
            'epoch_frac': loop.epoch_num + loop.batch_num / loop.n_batches,
            'batch_num' : loop.batch_num,
            'accuracy': loop.accuracy.item(),
            'loss': loop.loss.item(),
            'lr': loop.optimizer.param_groups[0]['lr'],
            'momentum': loop.optimizer.param_groups[0]['betas'][0],
        })


class LoopProgressBar(LoopComponent):

    def __init__(self, item_names=['loss', 'accuracy']):
        self.item_names = item_names

    def on_epoch_begin(self, loop):
        self.fp = 0
        self.fn = 0
        self.total, self.count = ({ name: 0.0 for name in self.item_names }, 0)
        self.pbar = tqdm(total=loop.n_batches, desc=f"{loop.epoch_desc} epoch {loop.epoch_num}")

    def on_batch_end(self, loop):
        n = len(loop.batch)
        self.count += n
        for name in self.item_names:
            self.total[name] += getattr(loop, name).item() * n
        self.pbar.update(1)
        if (not loop.is_training):
            means = { f'mean_{name}': self.total[name] / self.count for name in self.item_names }
            #means['P1'] = loop.precision
            #means['R1'] = loop.recall
            for i in range(len(_stoi)):
                means[f'F1_{i}'] = loop.f1s[i]
            self.pbar.set_postfix(means)
        else:
            means = { f'mean_{name}': self.total[name] / self.count for name in self.item_names }
            for i in range(len(_stoi)):
                means[f'F1_{i}'] = loop.f1s[i]
            self.pbar.set_postfix(means)

    def on_epoch_end(self, loop):
        self.pbar.close()

#Initialize model.
n_classes = len(_stoi)
print("Num classes: " + str(n_classes))
model = SSTClassifier(
    d_depth=mw.model.hparams.n_layer + 1,
    d_emb=mw.model.hparams.n_hidden,
    d_inp=64,
    d_cap=2,
    n_parts=64,
    n_classes=n_classes,
    n = MAX_LEN,
)
model = model.cuda(device=DEVICE)
print('Total number of parameters: {:,}'.format(sum(np.prod(p.shape) for p in model.parameters())))

trn_batches = load_xml('train_v1.4.xml', resample=[1,1,2])#, resample=[2,1,5])#,resample = [2, 1, 7])
val_batches = load_xml('dev_v1.4.xml')

class IterB:
    def __init__(self, inputs, batch_size = 32):
        self.inputs = inputs
        self.batch_size = batch_size
        self.indcs = np.arange(len(inputs))
        self.ni = -1
        self.ln = len(self.inputs) 
        self.ln = self.ln // self.batch_size + (0 if self.ln % self.batch_size == 0 else 1) 
    def __len__(self):
        return self.ln
    def __iter__(self):
        np.random.shuffle(self.indcs)
        self.ni = 0
        return self
    def __next__(self):
        if self.ni >= len(self.inputs):
            raise StopIteration
        batch = create_batch()
        for i in range(self.batch_size):
            tokens, mask, label = self.inputs[self.indcs[self.ni]]
            batch.text[0].append(mask)
            batch.text[1].append(tokens)
            batch.label.append(label)
            self.ni += 1
            if self.ni >= len(self.inputs):
                break
        
        batch.text = (torch.tensor(batch.text[0]).to(device=DEVICE), mw.get_hidden(torch.tensor(batch.text[1]).to(device=DEVICE)))
        batch.label = torch.tensor(batch.label).to(device=DEVICE)
        return batch

loop = TrainTestLoop(model, [LoopMain(n_classes, DEVICE), LoopProgressBar()], IterB(trn_batches), IterB(val_batches))
#model.load_state_dict(torch.load('result3.pt'))
#model.eval()

# Train model
loop.train(n_epochs=5)

torch.save(model.state_dict(), 'model64_.pt')

#model.load_state_dict(torch.load('result.pt'))
#model.eval()

tst_batches = load_xml('test_TIMESTAMP1.xml')
loop.test(IterB(tst_batches, batch_size = 16))

tst_batches = load_xml('test_TIMESTAMP2.xml')
loop.test(IterB(tst_batches, batch_size = 16))
