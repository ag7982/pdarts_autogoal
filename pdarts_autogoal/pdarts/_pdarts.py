import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from autogoal.grammar import *
from autogoal.utils import nice_repr

import time
import copy

from .utils import AvgrageMeter
from .genotypes import Genotype
from .operations import PRIMITIVES
from .model_search import Network

@nice_repr
class PDarts:
    """
    NAS Algorithm
    """

    def __init__(
        self,
        optimizer: CategoricalValue("sgd", "adam", "rmsprop"),
        momentum: ContinuousValue(0.0, 1.0),
        weight_decay: ContinuousValue(3e-10, 3e-3),
        arch_weight_decay: ContinuousValue(3e-10, 3e-3),
        epochs: DiscreteValue(10, 25),
        warmup_epochs: DiscreteValue(5, 10),
        init_channels: DiscreteValue(8, 32),
        layers: DiscreteValue(2, 3),
        grad_clip: ContinuousValue(-1.0, 1.0) ,
        dropout_rate: ContinuousValue(0.0, 0.8),
        arch_dropout_rate: ContinuousValue(0.0, 1.0),
        learning_rate: ContinuousValue(0.0001, 0.1),
        arch_learning_rate: ContinuousValue(0.0001, 0.1),
        learning_rate_min: ContinuousValue(0.0001, 0.01),
    ):
        self.has_cuda = torch.cuda.is_available()
        self.device = (
            torch.device("cuda") if self.has_cuda else torch.device("cpu")
        )

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.switches_normal = [ [True for _ in PRIMITIVES] for _ in range(14)]
        self.switches_reduce = [ [True for _ in PRIMITIVES] for _ in range(14)]

        # Hyperparameters
        self.optimizer_type = CategoricalValue("sgd", "adam", "rmsprop"),
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.arch_weight_decay = arch_weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.init_channels = init_channels
        self.layers = layers
        self.grad_clip = grad_clip
        self.dropout_rate = dropout_rate
        self.arch_dropout_rate = arch_dropout_rate
        self.learning_rate = learning_rate
        self.arch_learning_rate = arch_learning_rate
        self.learning_rate_min = learning_rate_min

    @staticmethod
    def _accuracy_topk(output, target, topk=(1,)):
        with torch.no_grad():
            _, pred = output.topk(max(topk), 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))


            return [
                correct[:k].contiguous().view(-1).float().sum(0).mul_(100.0/target.size(0))
                for k in topk
            ]

    @staticmethod
    def _get_min_k(input_in, k):
        input_in = copy.deepcopy(input_in)
        index = []
        for i in range(k):
            idx = np.argmin(input_in)
            index.append(idx)
            input_in[idx] = 1
    
        return index

    @staticmethod
    def _get_min_k_no_zero(w_in, idxs, k):
        w = copy.deepcopy(w_in)
        index = []
        zf = bool(0 in idxs) 
        if zf:
            w = w[1:]
            index.append(0)
            k -= 1
        for i in range(k):
            idx = np.argmin(w)
            w[idx] = 1
            if zf:
                idx += 1
            index.append(idx)
        return index

    def _get_optim(self, arch_parameters) -> Optimizer:
        if self.optimizer_type == "adam":
            return Adam(
                arch_parameters,
                lr=self.arch_learning_rate, 
                betas=(0.5, 0.999), 
                weight_decay=self.arch_weight_decay
            )
        elif self.optimizer_type == "sgd":
            return SGD(
                arch_parameters,
                lr=self.arch_learning_rate, 
                momentum=self.arch_momentum,
                weight_decay=self.arch_weight_decay
            )
        else:
            return RMSprop(
                arch_parameters,
                lr=self.arch_learning_rate, 
                weight_decay=self.arch_weight_decay
            )

    def _run_inner(self, train_loader, valid_loader, classes, add_layer):
        model = Network(
            self.init_channels, 
            classes, 
            self.layers + add_layer,
            self.criterion, 
            switches_normal=self.switches_normal, 
            switches_reduce=self.switches_reduce, 
            probability=self.dropout_rate
        )
        model = model.to(self.device)
        network_params = [
            v
            for k, v in model.named_parameters()
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce'))
        ]
   
        optimizer_sgd = SGD(
            network_params,
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        optimizer_arch = self._get_optim(model.arch_parameters())

        scheduler = CosineAnnealingLR(
            optimizer_sgd, 
            float(self.epochs), 
            eta_min=self.learning_rate_min
        )
    
        for epoch in range(self.epochs):
            epoch_start = time.time()
            # training
            if epoch < self.warmup_epochs:
                model.update_probability(
                    self.dropout_rate * (self.epochs - epoch - 1) / self.epochs
                )
                train_acc, train_obj = self._train(
                    train_loader, 
                    valid_loader, 
                    model, 
                    network_params, 
                    optimizer_sgd, 
                    optimizer_arch, 
                    train_arch=False
                )
            else:
                model.update_probability(
                    min(self.dropout_rate * np.exp(-(epoch - self.warmup_epochs) * 0.2), 1.0)
                )                
                train_acc, train_obj = self._train(
                    train_loader, 
                    valid_loader, 
                    model,
                    network_params,
                    optimizer_sgd,
                    optimizer_arch, 
                    train_arch=True
                )
            scheduler.step()
            epoch_duration = time.time() - epoch_start
            print(f"{epoch_duration=}: {epoch=} {train_acc=}")
        return model

    def _set_switches(self, arch_param, num_to_drop):
        # drop operations with low architecture weights
        normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
      
        for i in range(14):
            idxs_normal, idxs_reduce = [], []
            for j in range(len(PRIMITIVES)):
                if self.switches_normal[i][j]:
                    idxs_normal.append(j)
                if self.switches_reduce[i][j]:
                    idxs_reduce.append(j)
            for idx in self._get_min_k(normal_prob[i, :], num_to_drop):
                self.switches_normal[i][idxs_normal[idx]] = False

            for idx in self._get_min_k(reduce_prob[i, :], num_to_drop):
                self.switches_reduce[i][idxs_reduce[idx]] = False
        
    def _on_last(self, arch_param, num_to_drop):
        normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
      
#         for i in range(14):
#             idxs_normal, idxs_reduce = [], []
#             for j in range(len(PRIMITIVES)):
#                 if self.switches_normal[i][j]:
#                     idxs_normal.append(j)
#                 if self.switches_reduce[i][j]:
#                     idxs_reduce.append(j)
#             print(idxs_normal)
#             for idx in self._get_min_k(normal_prob[i, :], num_to_drop):
#                 self.switches_normal[i][idxs_normal[idx]] = False

#             for idx in self._get_min_k(reduce_prob[i, :], num_to_drop):
#                 self.switches_reduce[i][idxs_reduce[idx]] = False

        switches_normal = copy.deepcopy(self.switches_normal)
        switches_reduce = copy.deepcopy(self.switches_reduce)

        normal_final = [0 for idx in range(14)]
        reduce_final = [0 for idx in range(14)]

        # remove all Zero operations
        for i in range(14):
            if switches_normal[i][0]:
                normal_prob[i][0] = 0
            normal_final[i] = max(normal_prob[i])
            if switches_reduce[i][0]:
                reduce_prob[i][0] = 0
            reduce_final[i] = max(reduce_prob[i])                
        # Generate Architecture, similar to DARTS
        keep_normal = [0, 1]
        keep_reduce = [0, 1]
        n = 3
        start = 2
        for i in range(3):
            end = start + n
            tbsn = normal_final[start:end]
            tbsr = reduce_final[start:end]
            edge_n = sorted(range(n), key=lambda x: tbsn[x])
            keep_normal.append(edge_n[-1] + start)
            keep_normal.append(edge_n[-2] + start)
            edge_r = sorted(range(n), key=lambda x: tbsr[x])
            keep_reduce.append(edge_r[-1] + start)
            keep_reduce.append(edge_r[-2] + start)
            start = end
            n += 1
        # set switches according the ranking of arch parameters
        for i in range(14):
            if not i in keep_normal:
                for j in range(len(PRIMITIVES)):
                    self.switches_normal[i][j] = False
            if not i in keep_reduce:
                for j in range(len(PRIMITIVES)):
                    self.switches_reduce[i][j] = False   

    def _train(self, train_queue, valid_queue, model, network_params, optimizer, optimizer_arch, train_arch=True):
        objs, top1, top5 = AvgrageMeter(), AvgrageMeter(), AvgrageMeter()
        for step, (inputs, target) in enumerate(train_queue):
            model.train()
            n = inputs.size(0)
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            if train_arch:
                try:
                    input_search, target_search = next(valid_queue_iter)
                except:
                    valid_queue_iter = iter(valid_queue)
                    input_search, target_search = next(valid_queue_iter)
                input_search = input_search.to(self.device)
                target_search = target_search.to(self.device)
                optimizer_arch.zero_grad()
                loss_arch = self.criterion(
                    model(input_search), target_search
                )
                loss_arch.backward()
                nn.utils.clip_grad_norm_(model.arch_parameters(), self.grad_clip)
                optimizer_arch.step()
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = self.criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(network_params, self.grad_clip)
            optimizer.step()

            prec1, prec5 = self._accuracy_topk(logits, target, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

        return top1.avg, objs.avg
    
    def _infer(self, valid_queue, model):
        objs, top1, top5 = AvgrageMeter(), AvgrageMeter(), AvgrageMeter()

        model.eval()

        for step, (inputs, target) in enumerate(valid_queue):
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                logits = model(inputs)
                loss = self.criterion(logits, target)

            prec1, prec5 = self._accuracy_topk(logits, target, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

        return top1.avg, objs.avg

    def _parse_network(self):

        def _parse_switches(switches):
            n = 2
            start = 0
            gene = []
            step = 4
            for i in range(step):
                end = start + n
                for j in range(start, end):
                    for k in range(len(switches[j])):
                        if switches[j][k]:
                            gene.append((PRIMITIVES[k], j - start))
                start = end
                n = n + 1
            return gene

        concat = range(2, 6)
        
        return Genotype(
            normal=_parse_switches(self.switches_normal), normal_concat=concat, 
            reduce=_parse_switches(self.switches_reduce), reduce_concat=concat
        )

    def fit(self, train_loader, valid_loader):
        add_layer = [0, 1, 2]
        drop_layer = [3, 2, 2]

        for add_l, drop_l in zip(add_layer, drop_layer):
            model = self._run_inner(
                train_loader,
                valid_loader,
                len(train_loader.dataset.classes),
                add_l
            )
            self._set_switches(model.arch_parameters(), drop_l)
        
        self._on_last(model.arch_parameters(), drop_l)

        print(self._parse_network())
        
        return self._infer(valid_loader, model)