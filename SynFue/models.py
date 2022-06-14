import torch
import numpy as np
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from SynFue import sampling
from SynFue import util
from SynFue import Encoder
from SynFue import cross_attn


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


def get_head_tail_rep(h, head_tail_index):
    """

    :param h: torch.tensor [batch size, seq_len, feat_dim]
    :param head_tail_index: [batch size, term_num, 2]
    :return:
    """
    res = []
    batch_size = head_tail_index.size(0)
    term_num = head_tail_index.size(1)
    for b in range(batch_size):
        temp = []
        for t in range(term_num):
            temp.append(torch.index_select(h[b], 0, head_tail_index[b][t]).view(-1))
        res.append(torch.stack(temp, dim=0))
    res = torch.stack(res)
    return res


class SynFueBERT(BertPreTrainedModel):
    """ Span-based model to jointly extract terms and relations """

    VERSION = '1.2'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, term_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, args, max_pairs: int = 100,
                 beta: float = 0.3, alpha: float = 1.0, sigma: float = 1.0):
        super(SynFueBERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self.bert_dropout = nn.Dropout(args.bert_dropout)
        self.SynFue = Encoder.SynFueEncoder(opt=args)
        self.cc = cross_attn.CA_module(config.hidden_size, config.hidden_size, 1, dropout=1.0)

        # layers
        self.rel_classifier = nn.Linear(config.hidden_size * 4 + size_embedding * 2, relation_types)
        self.rel_classifier3 = nn.Linear(config.hidden_size * 3 + size_embedding * 3, relation_types)
        self.term_linear = nn.Linear(config.hidden_size * 7 + size_embedding, config.hidden_size)
        self.term_classifier = nn.Linear(config.hidden_size, term_types)
        self.dep_linear = nn.Linear(config.hidden_size, relation_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._term_types = term_types
        self._max_pairs = max_pairs
        self._beta = beta
        self._alpha = alpha
        self._sigma = sigma

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    # def init_weights(self):
    #     for name, param in self.SynFue.named_parameters():
    #         if name.find('weight') != -1:
    #             torch.nn.init.normal_(param, 0, 1)
    #         elif name.find('bias') != -1:
    #             torch.nn.init.constant_(param, 0)
    #         else:
    #             torch.nn.init.xavier_normal_(param)
    #     for name, param in self.cc.named_parameters():
    #         if name.find('weight') != -1:
    #             torch.nn.init.normal_(param, 0, 1)
    #         elif name.find('bias') != -1:
    #             torch.nn.init.constant_(param, 0)
    #         else:
    #             torch.nn.init.xavier_normal_(param)

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, term_masks: torch.tensor,
                       term_sizes: torch.tensor, term_spans: torch.tensor, term_types: torch.tensor,
                       relations: torch.tensor, rel_masks: torch.tensor,
                       simple_graph: torch.tensor, graph: torch.tensor,
                       relations3: torch.tensor, rel_masks3: torch.tensor, pair_mask: torch.tensor,
                       pos: torch.tensor = None, pieces2word: torch.tensor = None):
        """

        :param encodings: [B, L']
        :param context_masks: [B, L']
        :param term_masks: [B, max_term, L]
        :param term_sizes: [B, max_term]
        :param term_spans: [B, max_term, 2]
        :param term_types: [B, max_term]
        :param relations: [B, max_relation, 2]
        :param rel_masks: [B, max_relation, L]
        :param simple_graph: [B, L, L]
        :param graph: [B, L, L]
        :param relations3: [B, max_relation, 3]
        :param rel_masks3:[B, max_relation, max_relation, L]
        :param pair_mask: [B, max_relation]
        :param pos: [B, L]
        :param pieces2word: [B, L, L']
        :return:
        """
        # get contextualized token embeddings from last transformer layer
        word_reps, cls_output = self._bert_encoder(input_ids=encodings, pieces2word=pieces2word)
        h, dep_output = self.SynFue(word_reps=word_reps, simple_graph=simple_graph, graph=graph, pos=pos)

        batch_size = encodings.shape[0]

        # classify terms
        size_embeddings = self.size_embeddings(term_sizes)  # embed term candidate sizes
        term_clf, term_reps = self._classify_terms(h, term_masks, size_embeddings, cls_output)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)  # B, max_rel, L, H
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)  # B, max_rel, relation_types

        # get span representation
        # dep_output = [B, L, L, H] -> [batch size, span num, span num, feat_dim]
        span_repr, mapping_list = self.get_span_repr(term_spans, term_types, dep_output)
        # apply across-attention to compute the syntax-aware inter-span representation
        cross_attn_span = self.cc(span_repr)  # batch size, span_num, soan_num, feat_dim

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits, chunk_rel_clf3, chunk_dep_score = self._classify_relations(cross_attn_span,
                                                                                         term_reps,
                                                                                         size_embeddings,
                                                                                         relations, rel_masks,
                                                                                         h_large, i,
                                                                                         relations3, rel_masks3,
                                                                                         pair_mask, mapping_list)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            chunk_rel_clf = self._alpha * chunk_rel_clf + self._beta * chunk_rel_clf3 + self._sigma * chunk_dep_score
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        max_clf = torch.full_like(rel_clf, torch.max(rel_clf).item())
        min_clf = torch.full_like(rel_clf, torch.min(rel_clf).item())
        inifite = torch.full_like(rel_clf, 1e-18)
        rel_clf = torch.div(rel_clf - min_clf + inifite, max_clf - min_clf + inifite)

        return term_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, term_masks: torch.tensor,
                      term_sizes: torch.tensor, term_spans: torch.tensor, term_sample_masks: torch.tensor,
                      simple_graph: torch.tensor, graph: torch.tensor, pos: torch.tensor = None,
                      pieces2word: torch.tensor = None):
        # get contextualized token embeddings from last transformer layer
        # context_masks = context_masks.float()
        word_reps, cls_output = self._bert_encoder(input_ids=encodings, pieces2word=pieces2word)
        h, dep_output = self.SynFue(word_reps=word_reps, simple_graph=simple_graph, graph=graph, pos=pos)

        batch_size = encodings.shape[0]
        ctx_size = term_masks.shape[-1]

        # classify terms
        size_embeddings = self.size_embeddings(term_sizes)  # embed term candidate sizes
        term_clf, term_reps = self._classify_terms(h, term_masks, size_embeddings, cls_output)

        # ignore term candidates that do not constitute an actual term for relations (based on classifier)
        relations, rel_masks, rel_sample_masks, relations3, rel_masks3, \
        rel_sample_masks3, pair_mask, span_repr, mapping_list = self._filter_spans(term_clf, term_spans,
                                                                                   term_sample_masks,
                                                                                   ctx_size, dep_output)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        # h = self.rel_bert(input_ids=encodings, attention_mask=context_masks)[0]
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # get span representation
        cross_attn_span = self.cc(span_repr)  # batch size, seq_len, seq_len, feat_dim

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits, chunk_rel_clf3, chunk_dep_score = self._classify_relations(cross_attn_span,
                                                                                         term_reps,
                                                                                         size_embeddings,
                                                                                         relations, rel_masks,
                                                                                         h_large, i,
                                                                                         relations3, rel_masks3,
                                                                                         pair_mask, mapping_list)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            chunk_rel_clf = self._alpha * chunk_rel_clf + self._beta * chunk_rel_clf3 + self._sigma * chunk_dep_score
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        max_clf = torch.full_like(rel_clf, torch.max(rel_clf).item())
        min_clf = torch.full_like(rel_clf, torch.min(rel_clf).item())
        inifite = torch.full_like(rel_clf, 1e-18)
        rel_clf = torch.div(rel_clf - min_clf + inifite, max_clf - min_clf + inifite)

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        term_clf = torch.softmax(term_clf, dim=2)

        return term_clf, rel_clf, relations

    def _classify_terms(self, h, term_masks, size_embeddings, cls_output):
        """

        :param h: [b, L, H*2]
        :param term_masks: [B, max_term, L]
        :param size_embeddings: [b, max_term, size_embedding]
        :param cls_output:  [B, H]
        :return:
            term_clf: [B, max_term, term_types]
            term_spans_pool: [B, max_term, H]
        """
        # max pool term candidate spans
        # m = (term_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # term_spans_pool = m + h.unsqueeze(1).repeat(1, term_masks.shape[1], 1, 1)
        # term_spans_pool = term_spans_pool.max(dim=2)[0]

        min_value = torch.min(h).item()
        _h = h.unsqueeze(1).expand(-1, term_masks.size(1), -1, -1)
        _h = torch.masked_fill(_h, term_masks.eq(0).unsqueeze(-1), min_value)
        term_spans_pool, _ = torch.max(_h, dim=2)

        # get cls token as candidate context representation
        # term_ctx = get_token(h, encodings, self._cls_token)

        # get head and tail token representation
        m = term_masks.to(dtype=torch.long)
        k = torch.tensor(np.arange(0, term_masks.size(-1)), dtype=torch.long)
        k = k.unsqueeze(0).unsqueeze(0).repeat(term_masks.size(0), term_masks.size(1), 1).to(m.device)
        mk = torch.mul(m, k)  # element-wise multiply
        mk_max = torch.argmax(mk, dim=-1, keepdim=True)
        mk_min = torch.argmin(mk, dim=-1, keepdim=True)
        mk = torch.cat([mk_min, mk_max], dim=-1)
        head_tail_rep = get_head_tail_rep(h, mk)  # [batch size, term_num, bert_dim*2)

        # create candidate representations including context, max pooled span and size embedding, head and tail
        term_repr = torch.cat([cls_output.unsqueeze(1).repeat(1, term_spans_pool.shape[1], 1),
                               term_spans_pool, size_embeddings, head_tail_rep], dim=2)
        term_repr = self.dropout(term_repr)
        term_repr = self.term_linear(term_repr)

        # classify term candidates
        term_clf = self.term_classifier(term_repr)

        return term_clf, term_repr

    def _classify_relations(self, spans_matrix, term_spans_repr, size_embeddings, relations, rel_masks,
                            h, chunk_start, relations3, rel_masks3, pair_mask, rel_to_span):
        """

        :param spans_matrix:  [B, max_term, max_term, H]
        :param term_spans_repr: [B, max_term, H]
        :param size_embeddings: [B, max_term, term_size]
        :param relations: [B, max_rel, 2]
        :param rel_masks: [B, max_rel, L]
        :param h:  [B, max_rel, L, H*2]
        :param chunk_start:
        :param relations3: [B, max_rel, 3]
        :param rel_masks3: [B, max_rel, max_rel, L]
        :param pair_mask:
        :param rel_to_span:
        :return:
        """
        batch_size = relations.shape[0]
        feat_dim = spans_matrix.size(-1)

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        def get_span_idx(mapping_list, idx1, idx2):
            for x in mapping_list:
                if idx1 == x[0][0] and idx2 == x[0][1]:
                    return x[1][0], x[1][1]

        batch_dep_score = []
        for i in range(batch_size):
            _rel = relations[i]
            dep_score_list = []
            r_2_s = rel_to_span[i]
            for r in _rel:
                i1, i2 = r[0].item(), r[1].item()
                idx1, idx2 = get_span_idx(r_2_s, i1, i2)
                try:
                    feat = spans_matrix[i][idx1][idx2]
                except:
                    print('Out of bundary', spans_matrix.size(), i, i1, i2)
                    feat = torch.zeros(feat_dim)
                dep_socre = self.dep_linear(feat).item()
                dep_score_list.append([dep_socre])
            batch_dep_score.append(dep_score_list)

        batch_dep_score = torch.sigmoid(
            torch.tensor(batch_dep_score).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        # get pairs of term candidate representations
        term_pairs = util.batch_index(term_spans_repr, relations)
        term_pairs = term_pairs.view(batch_size, term_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between term candidate pair)
        # mask non term candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent term candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled term candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, term_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)
        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)

        if relations3.shape[1] > self._max_pairs:
            relations3 = relations3[:, chunk_start:chunk_start + self._max_pairs]
            # rel_masks3 = rel_masks3[:, chunk_start:chunk_start + self._max_pairs]

        p_num = relations3.size(1)
        p_tris = relations3.size(2)

        relations3 = relations3.view(batch_size, -1, 3)

        # get three pairs candidata representations
        term_pairs3 = util.batch_index(term_spans_repr, relations3)
        term_pairs3 = term_pairs3.view(batch_size, term_pairs3.shape[1], -1)

        size_pair_embeddings3 = util.batch_index(size_embeddings, relations3)
        size_pair_embeddings3 = size_pair_embeddings3.view(batch_size, size_pair_embeddings3.shape[1], -1)

        rel_repr = torch.cat([term_pairs3, size_pair_embeddings3], dim=2)
        rel_repr = self.dropout(rel_repr)
        # classify relation candidates
        chunk_rel_logits3 = self.rel_classifier3(rel_repr)

        chunk_rel_clf3 = chunk_rel_logits3.view(batch_size, p_num, p_tris, -1)
        chunk_rel_clf3 = torch.sigmoid(chunk_rel_clf3)

        chunk_rel_clf3 = torch.sum(chunk_rel_clf3, dim=2)
        chunk_rel_clf3 = torch.sigmoid(chunk_rel_clf3)

        return chunk_rel_logits, chunk_rel_clf3, batch_dep_score

    def _filter_spans(self, term_clf, term_spans, term_sample_masks, ctx_size, token_repr):
        """
        according to the results of term type detection, we first filter the invalid term, i.e., only keeping aspect
        term and opinion term, and then we construct term pairs and term triplets for the following relations detection.
        :param term_clf: [B, max_term, term_types]
        :param term_spans: [B, max_term, 2]
        :param term_sample_masks:
        :param ctx_size: L
        :param token_repr: [B, L, H]
        :return:
        """
        batch_size = term_clf.shape[0]
        feat_dim = token_repr.size(-1)
        term_logits_max = term_clf.argmax(dim=-1) * term_sample_masks.long()  # get term type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        batch_relations3 = []
        batch_rel_masks3 = []
        batch_rel_sample_masks3 = []
        batch_pair_mask = []

        batch_span_repr = []
        batch_rel_to_span = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []
            rels3 = []
            rel_masks3 = []
            sample_masks3 = []

            span_repr = []
            rel_to_span = []

            # get spans classified as terms
            non_zero_indices = (term_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = term_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            pair_mask = []
            for idx1, (i1, s1) in enumerate(zip(non_zero_indices, non_zero_spans)):
                temp = []
                for idx2, (i2, s2) in enumerate(zip(non_zero_indices, non_zero_spans)):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)
                        p_rels3 = []
                        p_masks3 = []
                        for i3, s3 in zip(non_zero_indices, non_zero_spans):
                            if i1 != i2 and i1 != i3 and i2 != i3:
                                p_rels3.append((i1, i2, i3))
                                p_masks3.append(sampling.create_rel_mask3(s1, s2, s3, ctx_size))
                                sample_masks3.append(1)
                        if len(p_rels3) > 0:
                            rels3.append(p_rels3)
                            rel_masks3.append(p_masks3)
                            pair_mask.append(1)
                        else:
                            rels3.append([(i1, i2, 0)])
                            rel_masks3.append([sampling.create_rel_mask3(s1, s2, (0, 0), ctx_size)])
                            pair_mask.append(0)
                        rel_to_span.append([[i1, i2], [idx1, idx2]])
                    feat = \
                        torch.max(token_repr[i, s1[0]: s1[-1] + 1, s2[0]:s2[-1] + 1, :].contiguous().view(-1, feat_dim),
                                  dim=0)[0]
                    temp.append(feat)
                span_repr.append(temp)

            if not rels:
                # case: no more than two spans classified as terms
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
                batch_span_repr.append(torch.tensor([[[0] * feat_dim]], dtype=torch.float))
                batch_rel_to_span.append([[[0, 0], [0, 0]]])
            else:
                # case: more than two spans classified as terms
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))
                batch_span_repr.append(torch.stack([torch.stack(x) for x in span_repr]))
                batch_rel_to_span.append(rel_to_span)

            if not rels3:
                batch_relations3.append(torch.tensor([[[0, 0, 0]]], dtype=torch.long))
                batch_rel_masks3.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks3.append(torch.tensor([0], dtype=torch.bool))
                batch_pair_mask.append(torch.tensor([0], dtype=torch.bool))

            else:
                max_tri = max([len(x) for x in rels3])
                # print(max_tri)
                for idx, r in enumerate(rels3):
                    r_len = len(r)
                    if r_len < max_tri:
                        rels3[idx].extend([rels3[idx][0]] * (max_tri - r_len))
                        rel_masks3[idx].extend(
                            [rel_masks3[idx][0]] * (max_tri - r_len))
                batch_relations3.append(torch.tensor(rels3, dtype=torch.long))
                batch_rel_masks3.append(torch.stack([torch.stack(x) for x in rel_masks3]))
                batch_rel_sample_masks3.append(torch.tensor(sample_masks3, dtype=torch.bool))
                batch_pair_mask.append(torch.tensor(pair_mask, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)
        batch_span_repr = util.padded_stack(batch_span_repr).to(device)

        batch_relations3 = util.padded_stack(batch_relations3).to(device)
        batch_rel_masks3 = util.padded_stack(batch_rel_masks3).to(device)
        batch_rel_sample_masks3 = util.padded_stack(batch_rel_sample_masks3).to(device)
        batch_pair_mask = util.padded_stack(batch_pair_mask).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks, \
               batch_relations3, batch_rel_masks3, batch_rel_sample_masks3, \
               batch_pair_mask, batch_span_repr, batch_rel_to_span

    def get_span_repr(self, term_spans, term_types, token_repr):
        """

        :param term_spans: [B, span_num, 2]
        :param term_types: [B, span_num]
        :param token_repr: [B, L, L, feat_dim]
        :return:
            batch_span_repr: [B, span_num, span_num, feat_dim]
            batch_mapping_list: B. It is used to store the correspondence between the index in span matrix (x1, x2)
                                and the index in positive term span list (i1, i2).
        """
        batch_size = term_spans.size(0)
        feat_dim = token_repr.size(-1)
        batch_span_repr = []
        batch_mapping_list = []
        for i in range(batch_size):
            span_repr = []
            mapping_list = []
            # get target spans  as aspect term or opinion term
            non_zero_indices = (term_types[i] != 0).nonzero().view(-1)
            non_zero_spans = term_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()
            for x1, (i1, s1) in enumerate(zip(non_zero_indices, non_zero_spans)):
                temp = []
                for x2, (i2, s2) in enumerate(zip(non_zero_indices, non_zero_spans)):
                    feat = \
                        torch.max(token_repr[i, s1[0]: s1[-1] + 1, s2[0]:s2[-1] + 1, :].contiguous().view(-1, feat_dim),
                                  dim=0)[0]
                    temp.append(feat)
                    mapping_list.append([[i1, i2], [x1, x2]])

                span_repr.append(torch.stack(temp))
            batch_span_repr.append(torch.stack(span_repr))
            batch_mapping_list.append(mapping_list)

        device = self.rel_classifier.weight.device
        batch_span_repr = util.padded_stack(batch_span_repr).to(device)

        return batch_span_repr, batch_mapping_list

    def _bert_encoder(self, input_ids, pieces2word):
        """

        :param input_ids: [B, L'],  L' not equal L
        :param pieces2word: [B, L, L']
        :return:
            word_reps: [B, L, H]
            pooler_output: [B, H]
        """
        # sequence_output, pooled_output = self.bert(input_ids)
        bert_output = self.bert(input_ids=input_ids, attention_mask=input_ids.ne(0).float())
        sequence_output, pooler_output = bert_output[0], bert_output[1]
        bert_embs = self.bert_dropout(sequence_output)

        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        return word_reps, pooler_output

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'synfue': SynFueBERT,
}


def get_model(name):
    return _MODELS[name]
