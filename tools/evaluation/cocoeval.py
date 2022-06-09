

import os

import itertools
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
from collections import OrderedDict
from .logging import print_log
import copy
import datetime
import time
from collections import defaultdict

import numpy as np


class EvalByRecall(COCOeval):

    def __init__(self, areaReg=None, areaRegLbl=None, **kwargs):
        super().__init__(**kwargs)

        if areaReg is not None:
            assert areaRegLbl is not None, "Missing 'areaRegLbl'."
            self.params.areaRng = areaReg
            self.params.areaRngLbl = areaRegLbl


class EvalByScore(COCOeval):

    def __init__(self, score_thrs, areaReg=None, areaRegLbl=None, **kwargs):
        super().__init__(**kwargs)

        self.params.scoThrs = score_thrs if isinstance(score_thrs, list) else [score_thrs]

        if areaReg is not None:
            assert areaRegLbl is not None, "Missing 'areaRegLbl'."
            self.params.areaRng = areaReg
            self.params.areaRngLbl = areaRegLbl

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in
        self.eval

        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        S = len(p.scoThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, S, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, S, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                           inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((S, ))
                        re = np.zeros((S, ))

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        inds = len(dtScoresSorted) - np.searchsorted(dtScoresSorted[::-1], p.scoThrs, side='left') - 1
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                re[ri] = rc[pi]
                        except:  # noqa: E722
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        recall[t, :, k, a, m] = np.array(re)
        self.eval = {
            'params': p,
            'counts': [T, S, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
