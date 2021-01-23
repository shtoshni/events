from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]
        self.evaluators.append(BlancEvaluator())

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


class BlancEvaluator(object):
    def __init__(self, beta=1):
        self.right_coref = 0
        self.wrong_coref = 0
        self.right_non = 0
        self.wrong_non = 0
        self.total_gold_coref = 0
        self.total_non_gold_coref = 0
        self.metric = blanc
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        rc, wc, rn, wn, gc, gnc = self.metric(predicted, gold)

        self.right_coref += rc
        self.wrong_coref += wc
        self.right_non += rn
        self.wrong_non += wn

        self.total_gold_coref += gc
        self.total_non_gold_coref += gnc

    def get_f1(self):
        beta = self.beta

        c_prec, nc_prec = self.get_precision(details=True)
        c_recall, nc_recall = self.get_recall(details=True)

        if c_prec + c_recall > 0.0:
            fc = (1 + beta * beta) * c_prec * c_recall / (beta * beta * c_prec + c_recall)
        else:
            fc = 0.0

        if nc_prec + nc_recall > 0.0:
            fnc = (1 + beta * beta) * nc_prec * nc_recall / (beta * beta * nc_prec + nc_recall)
        else:
            fnc = 0.0

        return (fc + fnc)/2.0

    def get_recall(self, details=False):
        rc_recall = self.right_coref / self.total_gold_coref
        rn_recall = self.right_non / self.total_non_gold_coref

        if not details:
            return (rc_recall + rn_recall) / 2
        else:
            return rc_recall, rn_recall

    def get_precision(self, details=False):
        rc_prec = self.right_coref / (self.right_coref + self.wrong_coref + 1e-8)
        rn_prec = self.right_non / (self.right_non + self.wrong_non + 1e-8)

        if not details:
            return (rc_prec + rn_prec) / 2
        else:
            return rc_prec, rn_prec

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    # clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    # matching = linear_assignment(-scores)
    matching = linear_sum_assignment(-scores)
    matching = np.asarray(matching)
    matching = np.transpose(matching)

    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def blanc(predicted, gold):
    def get_coref_and_non_coref_links(clusters):
        coref_links = set()
        mentions = set()
        for cluster in clusters:
            for idx, mention1 in enumerate(cluster):
                mentions.add(tuple(mention1))
                for mention2 in cluster[idx + 1:]:
                    link = tuple(sorted([tuple(mention1), tuple(mention2)], key=lambda x: x[0] + 1e-5 * x[1]))
                    coref_links.add(link)

        non_coref_links = set()
        mentions = sorted(list(mentions), key=lambda x: x[0] + 1e-5 * x[1])
        for idx, mention1 in enumerate(mentions):
            for mention2 in mentions[idx + 1:]:
                if (mention1, mention2) in coref_links:
                    continue
                else:
                    non_coref_links.add((mention1, mention2))

        return coref_links, non_coref_links

    gold_cl, gold_noncl = get_coref_and_non_coref_links(gold)
    predicted_cl, predicted_noncl = get_coref_and_non_coref_links(predicted)

    rc, wc, rn, wn = 0, 0, 0, 0

    for mention_pair in predicted_cl:
        if mention_pair in gold_cl:
            rc += 1
        else:
            wc += 1

    for mention_pair in predicted_noncl:
        if mention_pair in gold_noncl:
            rn += 1
        else:
            wn += 1

    return rc, wc, rn, wn, len(gold_cl), len(gold_noncl)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem
