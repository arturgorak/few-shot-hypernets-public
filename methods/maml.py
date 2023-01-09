# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml
import os
import shutil
from copy import copy, deepcopy

import torch
import backbone
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from time import time
import torchvision.transforms as T


class MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx=False):
        super(MAML, self).__init__(model_func, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        # self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        # self.classifier.bias.data.fill_(0)

        self.maml_adapt_classifier = params.maml_adapt_classifier
        self.maml_adapt_classifier = True

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

        self.n_heads = 5
        self.module_list = nn.ModuleList()
        for idx in range(self.n_heads):
            self.module_list.append(backbone.Linear_fw(self.feat_dim, n_way))
            self.module_list[idx].bias.data.fill_(idx)

        self.changed_heads = 0
        self.current_head_index = 0
        self.prev_acc = 0

        self.scales_set = {}
        for x in range(self.n_heads):
            self.scales_set["Set " + str(x)] = 0


    def forward(self, x):
        x = nn.Dropout(p=0.2)(x)
        out = self.feature.forward(x)
        # scores = self.classifier.forward(out)
        scores = self.module_list[self.current_head_index].forward(out)
        return scores

    def set_forward(self, x, is_feature=False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support,
                                                                     *x.size()[2:])  # support data
        x_b_i = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,
                                                                     *x.size()[2:])  # query data
        y_a_i = Variable(
            torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()  # label for support data


        if self.maml_adapt_classifier:
            # fast_parameters = list(self.classifier.parameters())
            # for weight in self.classifier.parameters():
            #     weight.fast = None
            fast_parameters = list(self.module_list[self.current_head_index].parameters())
            for weight in self.module_list[self.current_head_index].parameters():
                weight.fast = None
        else:
            fast_parameters = list(
                self.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None

        self.zero_grad()

        for task_step in (list(range(self.task_update_num))):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn(scores, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters,
                                       create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            # parameters = self.classifier.parameters() if self.maml_adapt_classifier else self.parameters()
            parameters = self.module_list[self.current_head_index].parameters()

            for k, weight in enumerate(parameters):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[
                        k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(
                    weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
        query_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, query_data_labels)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100

        return loss, task_accuracy


    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print("Zaczynam się uczyć")
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        acc_all = []
        optimizer.zero_grad()

        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            loss, task_accuracy = self.set_forward_loss(x)

            if task_accuracy < self.prev_acc:
                if self.n_heads - 1 > self.changed_heads:
                    self.changed_heads += 1
                    self.module_list[self.changed_heads] = deepcopy(self.module_list[self.current_head_index])
                    self.current_head_index = self.changed_heads
                else:
                    for idx in range(self.n_heads):
                        if idx != self.current_head_index:
                            prev_index = self.current_head_index
                            self.current_head_index = idx
                            loss2, task_accuracy2 = self.set_forward_loss(x)

                            if task_accuracy < task_accuracy2:
                                print("Zmieniłem zestaw wag na: " + str(self.current_head_index))
                                loss, task_accuracy = loss2, task_accuracy2
                            else:
                                self.current_head_index = prev_index
                                print("Zostawiłem zestaw wag na: " + str(self.current_head_index))

                print("Zakończyłem szukanie najlepszych wag")

            self.scales_set["Set " + str(self.current_head_index)] += 1

            self.prev_acc = task_accuracy

            avg_loss = avg_loss + loss.item()  # .data[0]
            loss_all.append(loss)
            acc_all.append(task_accuracy)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []

            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        metrics = {"accuracy/train": acc_mean, "Prev acc": self.prev_acc}
        metrics.update(self.scales_set)

        return metrics

    def test_loop(self, test_loader, return_std=False, return_time: bool = False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []
        eval_time = 0
        iter_num = len(test_loader)
        print(f"Jest {self.n_heads} zestawów wag")
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            s = time()

            correct_this, count_this, certainty = self.correct(x)

            for idx in range(self.n_heads):
                if self.current_head_index != idx:
                    prev_idx = self.current_head_index
                    self.current_head_index = idx
                    correct_this2, count_this2, certainty2 = self.correct(x)

                    if certainty < certainty2:
                        correct_this, count_this, certainty = correct_this2, count_this2, certainty2
                        print(f"Zmieniłem zestaw wag na zestaw {self.current_head_index}")
                    else:
                        self.current_head_index = prev_idx
                        print(f"Zostawiłem zestaw wag na zestaw {self.current_head_index}")
            print("Zakończyłem wybieranie najlepszego zestawu wag")

            t = time()
            eval_time += (t - s)
            acc_all.append(correct_this / count_this * 100)

        num_tasks = len(acc_all)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        if return_std:
            ret.append(acc_std)
        if return_time:
            ret.append(eval_time)
        ret.append({})

        return ret

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits = self.set_forward(x)
        return logits

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        # topk_values = topk_scores.sum(dim=0).cpu().numpy()
        # top1_confidence = topk_scores[:, 0].cpu() / topk_values

        m = torch.nn.Softmax(dim=0)
        output = float(max(m(topk_scores)))

        return float(top1_correct), len(y_query), output # tutaj zmieniłem