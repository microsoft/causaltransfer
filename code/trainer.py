import pdb
import torch
from torch import nn
from transformers import Trainer
from itertools import combinations
from utils import compute_irm_penalty, compute_mmd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IRMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        clusters = inputs['labels'][:,1]
        inputs['labels'] = inputs['labels'][:, 0].to(torch.long)
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()

        unique_clusters = list(set(clusters))
        penalty = 0
        loss = 0
        penalty_multiplier = self.state.epoch ** 1.6

        for cluster in unique_clusters:
            dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE).requires_grad_()
            logits_cluster = outputs['logits'][clusters == cluster]
            labels_cluster = inputs['labels'][clusters == cluster]
            loss_cluster = loss_fct(logits_cluster.view(-1, 2)*dummy_w, labels_cluster.view(-1))
            if self.control.should_evaluate:
                penalty = 1.0
            else:
                penalty += compute_irm_penalty(loss_cluster, dummy_w)
            loss += loss_cluster

        loss += penalty_multiplier * penalty
        if penalty_multiplier > 1.0:
            loss /= penalty_multiplier

        return (loss, outputs) if return_outputs else loss

class CITrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        clusters = inputs['labels'][:,1]
        inputs['labels'] = inputs['labels'][:,0].to(torch.long)
        loss_fct = nn.CrossEntropyLoss()
        outputs = model(**inputs)
        logits = outputs['logits']
        labels = inputs['labels']
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        unique_clusters = list(set(clusters))
        mmd_loss = 0

        penalty_multiplier = self.state.epoch ** 1.6

        for comb in combinations(unique_clusters, 2):
            cluster1 = comb[0]
            cluster2 = comb[1]

            logits_cluster1 = outputs['logits'][clusters == cluster1]
            logits_cluster2 = outputs['logits'][clusters == cluster2]
            
            mmd_loss += compute_mmd(logits_cluster1, logits_cluster2)

        loss += penalty_multiplier * mmd_loss
        return (loss, outputs) if return_outputs else loss