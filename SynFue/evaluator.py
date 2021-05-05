import json
import os
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import xBertTokenizer

from SynFue import util
from SynFue.terms import Document, Dataset, TermType
from SynFue.input_reader import JsonInputReader
from SynFue.opt import jinja2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, no_overlapping: bool,
                 predictions_path: str, examples_path: str, example_count: int, epoch: int, dataset_label: str):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold
        self._no_overlapping = no_overlapping

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        self._examples_path = examples_path
        self._example_count = example_count

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # prediction

        # terms
        self._gt_terms = []  # ground truth
        self._pred_terms = []  # prediction

        self._pseudo_term_type = TermType('Term', 1, 'Term', 'Term')  # for span only evaluation

        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_term_clf: torch.tensor, batch_rel_clf: torch.tensor,
                   batch_rels: torch.tensor, batch: dict):
        batch_size = batch_rel_clf.shape[0]
        rel_class_count = batch_rel_clf.shape[2]
        # get maximum activation (index of predicted term type)
        batch_term_types = batch_term_clf.argmax(dim=-1)
        # apply term sample mask
        batch_term_types *= batch['term_sample_masks'].long()

        batch_rel_clf = batch_rel_clf.view(batch_size, -1)

        # apply threshold to relations
        if self._rel_filter_threshold > 0:
            batch_rel_clf[batch_rel_clf < self._rel_filter_threshold] = 0

        for i in range(batch_size):
            # get model predictions for sample
            rel_clf = batch_rel_clf[i]
            term_types = batch_term_types[i]

            # get predicted relation labels and corresponding term pairs
            rel_nonzero = rel_clf.nonzero().view(-1)
            rel_scores = rel_clf[rel_nonzero]

            rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1)
            rel_indices = rel_nonzero // rel_class_count

            rels = batch_rels[i][rel_indices]

            # get masks of terms in relation
            rel_term_spans = batch['term_spans'][i][rels].long()

            # get predicted term types
            rel_term_types = torch.zeros([rels.shape[0], 2])
            if rels.shape[0] != 0:
                rel_term_types = torch.stack([term_types[rels[j]] for j in range(rels.shape[0])])

            # convert predicted relations for evaluation
            sample_pred_relations = self._convert_pred_relations(rel_types, rel_term_spans,
                                                                 rel_term_types, rel_scores)

            # get terms that are not classified as 'None'
            valid_term_indices = term_types.nonzero().view(-1)
            valid_term_types = term_types[valid_term_indices]
            valid_term_spans = batch['term_spans'][i][valid_term_indices]
            valid_term_scores = torch.gather(batch_term_clf[i][valid_term_indices], 1,
                                               valid_term_types.unsqueeze(1)).view(-1)
            sample_pred_terms = self._convert_pred_terms(valid_term_types, valid_term_spans,
                                                               valid_term_scores)

            if self._no_overlapping:
                sample_pred_terms, sample_pred_relations = self._remove_overlapping(sample_pred_terms,
                                                                                       sample_pred_relations)

            self._pred_terms.append(sample_pred_terms)
            self._pred_relations.append(sample_pred_relations)

    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Terms (named term recognition (NTR)) ---")
        print("An term is considered correct if the term type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_terms, self._pred_terms, include_term_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without named term classification (NTC)")
        print("A relation is considered correct if the relation type and the spans of the two "
              "related terms are predicted correctly (term type is not considered)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_term_types=False)
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With named term classification (NTC)")
        print("A relation is considered correct if the relation type and the two "
              "related terms are predicted correctly (in span and term type)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_term_types=True)
        rel_nec_eval = self._score(gt, pred, print_results=True)

        return ner_eval, rel_eval, rel_nec_eval

    def store_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            pred_terms = self._pred_terms[i]
            pred_relations = self._pred_relations[i]

            # convert terms
            converted_terms = []
            for term in pred_terms:
                term_span = term[:2]
                span_tokens = util.get_span_tokens(tokens, term_span)
                term_type = term[2].identifier
                # if term_type == 'None':
                #     continue
                converted_term = dict(type=term_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
                converted_terms.append(converted_term)
            converted_terms = sorted(converted_terms, key=lambda e: e['start'])

            # print('converted_terms: ', converted_terms)

            # convert relations
            converted_relations = []
            for relation in pred_relations:
                head, tail = relation[:2]
                head_span, head_type = head[:2], head[2].identifier
                tail_span, tail_type = tail[:2], tail[2].identifier
                head_span_tokens = util.get_span_tokens(tokens, head_span)
                tail_span_tokens = util.get_span_tokens(tokens, tail_span)
                relation_type = relation[2].identifier

                converted_head = dict(type=head_type, start=head_span_tokens[0].index,
                                      end=head_span_tokens[-1].index + 1)
                converted_tail = dict(type=tail_type, start=tail_span_tokens[0].index,
                                      end=tail_span_tokens[-1].index + 1)

                # print(converted_tail)
                head_idx = converted_terms.index(converted_head)
                tail_idx = converted_terms.index(converted_tail)

                converted_relation = dict(type=relation_type, head=head_idx, tail=tail_idx)
                converted_relations.append(converted_relation)
            converted_relations = sorted(converted_relations, key=lambda r: r['head'])

            doc_predictions = dict(tokens=[t.phrase for t in tokens], terms=converted_terms,
                                   relations=converted_relations)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        term_examples = []
        rel_examples = []
        rel_examples_nec = []

        for i, doc in enumerate(self._dataset.documents):
            # terms
            term_example = self._convert_example(doc, self._gt_terms[i], self._pred_terms[i],
                                                   include_term_types=True, to_html=self._term_to_html)
            term_examples.append(term_example)

            # relations
            # without term types
            rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_term_types=False, to_html=self._rel_to_html)
            rel_examples.append(rel_example)

            # with term types
            rel_example_nec = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                    include_term_types=True, to_html=self._rel_to_html)
            rel_examples_nec.append(rel_example_nec)

        label, epoch = self._dataset_label, self._epoch

        # terms
        self._store_examples(term_examples[:self._example_count],
                             file_path=self._examples_path % ('terms', label, epoch),
                             template='term_examples.html')

        self._store_examples(sorted(term_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('terms_sorted', label, epoch),
                             template='term_examples.html')

        # relations
        # without term types
        self._store_examples(rel_examples[:self._example_count],
                             file_path=self._examples_path % ('rel', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_sorted', label, epoch),
                             template='relation_examples.html')

        # with term types
        self._store_examples(rel_examples_nec[:self._example_count],
                             file_path=self._examples_path % ('rel_nec', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples_nec[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_nec_sorted', label, epoch),
                             template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations
            gt_terms = doc.terms

            # convert ground truth relations and terms for precision/recall/f1 evaluation
            sample_gt_terms = [term.as_tuple() for term in gt_terms]
            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]

            if self._no_overlapping:
                sample_gt_terms, sample_gt_relations = self._remove_overlapping(sample_gt_terms,
                                                                                   sample_gt_relations)

            self._gt_terms.append(sample_gt_terms)
            self._gt_relations.append(sample_gt_relations)

    def _convert_pred_terms(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            term_type = self._input_reader.get_term_type(label_idx)

            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()

            converted_pred = (start, end, term_type, score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _convert_pred_relations(self, pred_rel_types: torch.tensor, pred_term_spans: torch.tensor,
                                pred_term_types: torch.tensor, pred_scores: torch.tensor):
        converted_rels = []
        check = set()

        for i in range(pred_rel_types.shape[0]):
            label_idx = pred_rel_types[i].item()
            pred_rel_type = self._input_reader.get_relation_type(label_idx)
            pred_head_type_idx, pred_tail_type_idx = pred_term_types[i][0].item(), pred_term_types[i][1].item()
            pred_head_type = self._input_reader.get_term_type(pred_head_type_idx)
            pred_tail_type = self._input_reader.get_term_type(pred_tail_type_idx)
            score = pred_scores[i].item()

            spans = pred_term_spans[i]
            head_start, head_end = spans[0].tolist()
            tail_start, tail_end = spans[1].tolist()

            converted_rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type)
            converted_rel = self._adjust_rel(converted_rel)

            if converted_rel not in check:
                check.add(converted_rel)
                converted_rels.append(tuple(list(converted_rel) + [score]))

        return converted_rels

    def _remove_overlapping(self, terms, relations):
        non_overlapping_terms = []
        non_overlapping_relations = []

        for term in terms:
            if not self._is_overlapping(term, terms):
                non_overlapping_terms.append(term)

        for rel in relations:
            e1, e2 = rel[0], rel[1]
            if not self._check_overlap(e1, e2):
                non_overlapping_relations.append(rel)

        return non_overlapping_terms, non_overlapping_relations

    def _is_overlapping(self, e1, terms):
        for e2 in terms:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True

    def _adjust_rel(self, rel: Tuple):
        adjusted_rel = rel
        if rel[-1].symmetric:
            head, tail = rel[:2]
            if tail[0] < head[0]:
                adjusted_rel = tail, head, rel[-1]

        return adjusted_rel

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_term_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove term types based on setting
        def convert(t):
            if not include_term_types:
                # remove term type and score for evaluation
                if type(t[0]) == int:  # term
                    c = [t[0], t[1], self._pseudo_term_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_term_type),
                         (t[1][0], t[1][1], self._pseudo_term_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_term_types: bool, to_html):
        encoding = doc.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_term_types=include_term_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _term_to_html(self, term: Tuple, encoding: List[int]):
        start, end = term[:2]
        term_type = term[2].verbose_name

        tag_start = ' <span class="term">'
        tag_start += '<span class="type">%s</span>' % term_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
