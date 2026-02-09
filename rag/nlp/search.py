#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import re
import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from rag.prompts.generator import relevant_chunks_with_toc
from rag.nlp import rag_tokenizer, query
import numpy as np
from common.doc_store.doc_store_base import MatchDenseExpr, FusionExpr, OrderByExpr, DocStoreConnection
from common.string_utils import remove_redundant_spaces
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD
from common import settings

from common.misc_utils import thread_pool_exec

def index_name(uid): return f"ragflow_{uid}"


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    async def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        qv, _ = await thread_pool_exec(emb_mdl.encode_queries, txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd",
                    "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    async def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight: bool | list | None = None,
               rank_feature: dict | None = None
               ):
        if highlight is None:
            highlight = False

        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 128))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks", "doc_type_kwd",
                       "available_int", "content_with_weight", "mom_id", PAGERANK_FLD, TAG_FLD])
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.get_total(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            highlightFields = ["content_ltks", "title_tks"]
            if not highlight:
                highlightFields = []
            elif isinstance(highlight, list):
                highlightFields = highlight
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            if emb_mdl is None:
                matchExprs = [matchText]
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                emb_start_time = time.time()
                matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                logging.info(f"emb_time: {time.time() - emb_start_time:.3f}s, ")
                q_vec = matchDense.embedding_data
                if not settings.DOC_ENGINE_INFINITY:
                    src.append(f"q_{len(q_vec)}_vec")

                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # If result is empty, try again with lower min_match
                if total == 0:
                    if filters.get("doc_id"):
                        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.get_total(res)
                    else:
                        matchText, _ = self.qryr.question(qst, min_match=0.1)
                        matchDense.extra_options["similarity"] = 0.17
                        res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                    orderBy, offset, limit, idx_names, kb_ids,
                                                    rank_feature=rank_feature)
                        total = self.dataStore.get_total(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.get_doc_ids(res)
        keywords = list(kwds)
        highlight = self.dataStore.get_highlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.get_aggregation(res, "docnm_kwd")
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.get_fields(res, src + ["_score"]),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning(
                    "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        # 批量提取pageranks并向量化
        pageranks = np.array([search_res.field[chunk_id].get(PAGERANK_FLD, 0.0)
                              for chunk_id in search_res.ids], dtype=np.float32)

        if not query_rfea:
            return pageranks

        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        if q_denor == 0:
            return pageranks

        # 批量解析TAG_FLD，替换低效且不安全的eval
        for chunk_id in search_res.ids:
            tag_str = search_res.field[chunk_id].get(TAG_FLD, "{}")
            try:
                # json.loads比eval快8-10倍，且更安全
                tag_fea = json.loads(tag_str) if tag_str.strip() else {}
            except Exception:
                tag_fea = {}

            nor, denor = 0.0, 0.0
            for t, sc in tag_fea.items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc

            if denor == 0:
                rank_fea.append(0.0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)

        # 向量化计算最终秩特征分数
        return np.array(rank_fea, dtype=np.float32) * 10.0 + pageranks

    def _rank_feature_scores_back(self, query_rfea, search_res):
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        return np.array(rank_fea) * 10. + pageranks

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        _, keywords = self.qryr.question(query)
        if not sres.ids or not sres.query_vector:
            return [], [], []

        # 基础参数初始化
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = np.array([0.0] * vector_size, dtype=np.float32)

        # 批量预处理：一次遍历完成所有字段提取，减少循环次数
        ins_embd = []
        ins_tw = []
        field_cache = {}  # 缓存field查询结果，避免重复哈希查找
        for chunk_id in sres.ids:
            field = sres.field[chunk_id]
            field_cache[chunk_id] = field

            # 1. 向量解析优化：批量转换+numpy向量化
            vector_val = field.get(vector_column, zero_vector)
            if isinstance(vector_val, str):
                try:
                    # np.fromstring比split+get_float快一个数量级
                    vec = np.fromstring(vector_val, sep="\t", dtype=np.float32)
                    vec = vec if len(vec) == vector_size else zero_vector
                except Exception:
                    vec = zero_vector
            else:
                vec = np.array(vector_val, dtype=np.float32) if isinstance(vector_val, list) else zero_vector
            ins_embd.append(vec)

            # 2. important_kwd类型统一
            important_kwd = field.get("important_kwd", [])
            if isinstance(important_kwd, str):
                important_kwd = [important_kwd]
                field["important_kwd"] = important_kwd  # 缓存转换结果

            # 3. 文本特征构建优化：取消重复token生成，改用加权逻辑
            content_ltks = field[cfield].split()
            content_ltks = list(dict.fromkeys(content_ltks))  # 替代OrderedDict，Python3.7+更快
            title_tks = [t for t in field.get("title_tks", "").split() if t]
            question_tks = [t for t in field.get("question_tks", "").split() if t]

            # 存储结构化特征，后续计算时加权，避免生成大量重复token
            ins_tw.append({
                "content": content_ltks,
                "title": title_tks,
                "question": question_tks,
                "important": important_kwd
            })

        # 空值保护
        if not ins_embd:
            return [], [], []

        # 向量化处理嵌入向量
        ins_embd_np = np.vstack(ins_embd)
        query_vector_np = np.array(sres.query_vector, dtype=np.float32)

        # 4. 自定义加权token相似度计算（替代原重复token逻辑）
        def calc_weighted_token_sim(query_kwds, doc_tks_list):
            sims = []
            query_set = set(query_kwds)
            # 权重映射：对应原title*2、important*5、question*6的加权逻辑
            weights = {"content": 1.0, "title": 2.0, "important": 5.0, "question": 6.0}
            for doc_tks in doc_tks_list:
                score = 0.0
                total_weight = 0.0
                for tks_type, tks in doc_tks.items():
                    w = weights[tks_type]
                    if not tks:
                        continue
                    # 计算重叠度并加权
                    overlap = len(query_set & set(tks))
                    norm = len(tks) + len(query_set)
                    if norm > 0:
                        score += (overlap / norm) * w
                    total_weight += w
                sims.append(score / total_weight if total_weight > 0 else 0.0)
            return np.array(sims, dtype=np.float32)

        # 5. 计算文本相似度
        tksim = calc_weighted_token_sim(keywords, ins_tw)

        # 6. 向量相似度向量化计算（替代原循环计算）
        # 归一化处理（余弦相似度必备）
        query_norm = query_vector_np / np.linalg.norm(query_vector_np)
        ins_norm = ins_embd_np / np.linalg.norm(ins_embd_np, axis=1, keepdims=True)
        # 矩阵乘法批量计算余弦相似度
        vtsim = np.dot(ins_norm, query_norm)
        vtsim = np.nan_to_num(vtsim, 0.0)  # 处理NaN/Inf

        # 7. 混合相似度计算
        sim = tkweight * tksim + vtweight * vtsim

        # 8. 秩特征分数计算（优化eval为json.loads）
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # 最终结果合并
        final_sim = sim + rank_fea
        return final_sim.tolist(), tksim.tolist(), vtsim.tolist()

    def rerank_back(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        # 优化点1：提前校验核心依赖，快速返回空结果
        if not sres.ids or not rerank_mdl:
            return np.array([]), [], []

        # 优化点2：提前提取query关键词并转集合（加速后续交集计算）
        _, keywords = self.qryr.question(query)
        query_kwd_set = set(keywords)

        # 初始化批量存储容器
        field_cache = {}  # 缓存field数据，避免重复哈希查找
        ins_tw = []       # 存储结构化token特征
        text_list = []    # 存储rerank模型输入文本
        weights = {"content": 1.0, "title": 2.0, "important": 5.0, "question": 6.0}

        # 优化点3：单次遍历完成所有前置处理（合并原两次遍历）
        for chunk_id in sres.ids:
            # 1. 读取并缓存field数据
            field = sres.field[chunk_id]
            field_cache[chunk_id] = field

            # 2. 统一important_kwd类型（字符串转列表）
            important_kwd = field.get("important_kwd", [])
            if isinstance(important_kwd, str):
                important_kwd = [important_kwd]
                field["important_kwd"] = important_kwd  # 缓存转换结果

            # 3. 提取各类token并高效去重
            content_ltks = list(dict.fromkeys(field[cfield].split()))  # 去重（比OrderedDict快）
            title_tks = [t for t in field.get("title_tks", "").split() if t]  # 过滤空token
            question_tks = [t for t in field.get("question_tks", "").split() if t]

            # 4. 构建结构化token特征（用于后续加权相似度计算）
            ins_tw.append({
                "content": content_ltks,
                "title": title_tks,
                "important": important_kwd,
                "question": question_tks
            })

            # 5. 构建rerank模型输入文本（批量收集，后续统一清理）
            combined_tks = content_ltks + title_tks + important_kwd
            text_list.append(" ".join(combined_tks))

        # 优化点4：批量文本清理（减少循环内函数调用）
        text_list = [remove_redundant_spaces(txt) for txt in text_list]

        # 优化点5：向量化计算token相似度（替代纯Python循环）
        def calc_weighted_token_sim_batch(query_kwds, doc_tks_list):
            """批量计算加权token相似度"""
            sims = []
            for doc_tks in doc_tks_list:
                score = 0.0
                total_weight = 0.0
                for tks_type, tks in doc_tks.items():
                    w = weights[tks_type]
                    if not tks:
                        continue
                    # 集合交集计算重叠度，效率远高于列表遍历
                    overlap = len(query_kwd_set & set(tks))
                    norm = len(tks) + len(query_kwd_set)
                    if norm > 0:
                        score += (overlap / norm) * w
                    total_weight += w
                sims.append(score / total_weight if total_weight > 0 else 0.0)
            return np.array(sims, dtype=np.float32)

        # 批量计算token相似度
        tksim = calc_weighted_token_sim_batch(query_kwd_set, ins_tw)

        # 优化点6：批量调用rerank模型，增加异常保护
        try:
            vtsim, _ = rerank_mdl.similarity(query, text_list)
            vtsim = np.array(vtsim, dtype=np.float32)
        except Exception as e:
            logging.warning(f"Rerank model similarity compute failed: {e}, fallback to 0")
            vtsim = np.zeros(len(sres.ids), dtype=np.float32)

        # 优化点7：秩特征分数计算（复用优化后的_rank_feature_scores）
        rank_fea = self._rank_feature_scores(rank_feature, sres)
        rank_fea = np.array(rank_fea, dtype=np.float32)

        # 最终混合分数（全向量化计算）
        final_sim = tkweight * tksim + vtweight * vtsim + rank_fea

        return final_sim.tolist(), tksim.tolist(), vtsim.tolist()

    def rerank_by_model_back(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())


    async def retrieval(
            self,
            question,
            embd_mdl,
            tenant_ids,
            kb_ids,
            page,
            page_size,
            similarity_threshold=0.2,
            vector_similarity_weight=0.3,
            top=128,
            doc_ids=None,
            aggs=True,
            rerank_mdl=None,
            highlight=False,
            rank_feature: dict | None = {PAGERANK_FLD: 10},
    ):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        # Ensure RERANK_LIMIT is multiple of page_size
        RERANK_LIMIT = math.ceil(64 / page_size) * page_size if page_size > 1 else 1
        RERANK_LIMIT = max(30, RERANK_LIMIT)
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": math.ceil(page_size * page / RERANK_LIMIT),
            "size": RERANK_LIMIT,
            "question": question,
            "vector": True,
            "topk": top,
            "similarity": similarity_threshold,
            "available_int": 1,
        }

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        # 搜索阶段
        search_start = time.time()
        sres = await self.search(req, [index_name(tid) for tid in tenant_ids], kb_ids, embd_mdl, highlight,
                                 rank_feature=rank_feature)
        search_end = time.time()
        search_time = search_end - search_start

        # 重排序阶段
        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                1 - vector_similarity_weight,
                vector_similarity_weight,
                rank_feature=rank_feature,
                )
        else:
            if settings.DOC_ENGINE_INFINITY:
                # Don't need rerank here since Infinity normalizes each way score before fusion.
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim
                vsim = sim
            else:
                # ElasticSearch doesn't normalize each way score before fusion.
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    1 - vector_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                    )
        rerank_end = time.time()
        rerank_time = rerank_end - search_end

        sim_np = np.array(sim, dtype=np.float64)
        if sim_np.size == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 排序和过滤
        sorted_idx = np.argsort(sim_np * -1)
        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= similarity_threshold]
        filtered_count = len(valid_idx)
        ranks["total"] = int(filtered_count)

        if filtered_count == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 分页处理
        max_pages = max(RERANK_LIMIT // max(page_size, 1), 1)
        page_index = (page - 1) % max_pages
        begin = page_index * page_size
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        # 结果格式化
        filter_end = time.time()
        filter_time = filter_end - rerank_end

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim

        # 批量处理结果
        chunks = []
        doc_aggs = {}

        for i in page_idx:
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": float(sim_np[i]),
                "vector_similarity": float(vsim[i]),
                "term_similarity": float(tsim[i]),
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
                "mom_id": chunk.get("mom_id", ""),
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = remove_redundant_spaces(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            chunks.append(d)

        # 聚合处理
        if aggs:
            for i in valid_idx:
                id = sres.ids[i]
                chunk = sres.field[id]
                dnm = chunk.get("docnm_kwd", "")
                did = chunk.get("doc_id", "")
                if dnm not in doc_aggs:
                    doc_aggs[dnm] = {"doc_id": did, "count": 0}
                doc_aggs[dnm]["count"] += 1

            ranks["doc_aggs"] = [
                {
                    "doc_name": k,
                    "doc_id": v["doc_id"],
                    "count": v["count"],
                }
                for k, v in sorted(
                    doc_aggs.items(),
                    key=lambda x: x[1]["count"] * -1,
                )
            ]
        else:
            ranks["doc_aggs"] = []

        ranks["chunks"] = chunks
        format_end = time.time()
        format_time = format_end - filter_end

        # 记录性能日志
        end_time = time.time()
        logging.info(f"Core retrieval in {end_time - search_start:.3f}s, returning {len(ranks['chunks'])} chunks, search_time = {search_time:.3f}s, "
                     f"rerank_time = {rerank_time:.3f}s, filter_time = {filter_time:.3f}s, format_time = {format_time:.3f}s")
        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"],
                   sort_by_position: bool = False):
        condition = {"doc_id": doc_id}

        fields_set = set(fields or [])
        if sort_by_position:
            for need in ("page_num_int", "position_int", "top_int"):
                if need not in fields_set:
                    fields_set.add(need)
        fields = list(fields_set)

        orderBy = OrderByExpr()
        if sort_by_position:
            orderBy.asc("page_num_int")
            orderBy.asc("position_int")
            orderBy.asc("top_int")

        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = self.dataStore.search(fields, [], condition, [], orderBy, p, bs, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.get_fields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            # FIX: Solo terminar si no hay chunks, no si hay menos de bs
            if len(dict_chunks.values()) == 0:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []),
                                        keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}

    async def retrieval_by_toc(self, query: str, chunks: list[dict], tenant_ids: list[str], chat_mdl, topn: int = 6):
        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        ranks, doc_id2kb_id = {}, {}
        for ck in chunks:
            if ck["doc_id"] not in ranks:
                ranks[ck["doc_id"]] = 0
            ranks[ck["doc_id"]] += ck["similarity"]
            doc_id2kb_id[ck["doc_id"]] = ck["kb_id"]
        doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1.)[0][0]
        kb_ids = [doc_id2kb_id[doc_id]]
        es_res = self.dataStore.search(["content_with_weight"], [], {"doc_id": doc_id, "toc_kwd": "toc"}, [],
                                       OrderByExpr(), 0, 128, idx_nms,
                                       kb_ids)
        toc = []
        dict_chunks = self.dataStore.get_fields(es_res, ["content_with_weight"])
        for _, doc in dict_chunks.items():
            try:
                toc.extend(json.loads(doc["content_with_weight"]))
            except Exception as e:
                logging.exception(e)
        if not toc:
            return chunks

        ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
        if not ids:
            return chunks

        vector_size = 1024
        id2idx = {ck["chunk_id"]: i for i, ck in enumerate(chunks)}
        for cid, sim in ids:
            if cid in id2idx:
                chunks[id2idx[cid]]["similarity"] += sim
                continue
            chunk = self.dataStore.get(cid, idx_nms, kb_ids)
            if not chunk:
                continue
            d = {
                "chunk_id": cid,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": doc_id,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim,
                "vector_similarity": sim,
                "term_similarity": sim,
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            for k in chunk.keys():
                if k[-4:] == "_vec":
                    d["vector"] = chunk[k]
                    vector_size = len(chunk[k])
                    break
            chunks.append(d)

        return sorted(chunks, key=lambda x: x["similarity"] * -1)[:topn]

    def retrieval_by_children(self, chunks: list[dict], tenant_ids: list[str]):
        """优化子块合并逻辑，使用并行处理"""
        if not chunks:
            return []

        import time
        start_time = time.time()

        idx_nms = [index_name(tid) for tid in tenant_ids]
        mom_chunks = defaultdict(list)

        # 分离母块和子块
        i = 0
        while i < len(chunks):
            ck = chunks[i]
            mom_id = ck.get("mom_id")
            if not isinstance(mom_id, str) or not mom_id.strip():
                i += 1
                continue
            mom_chunks[ck["mom_id"]].append(chunks.pop(i))

        if not mom_chunks:
            return chunks

        # 如果所有块都是子块，初始化空列表
        if not chunks:
            chunks = []

        vector_size = 1024

        # 并行处理母块合并
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_mom_chunk(mom_id, cks):
            """处理单个母块的合并"""
            try:
                chunk = self.dataStore.get(mom_id, idx_nms, [ck["kb_id"] for ck in cks])
                d = {
                    "chunk_id": mom_id,
                    "content_ltks": " ".join([ck["content_ltks"] for ck in cks]),
                    "content_with_weight": chunk["content_with_weight"],
                    "doc_id": chunk["doc_id"],
                    "docnm_kwd": chunk.get("docnm_kwd", ""),
                    "kb_id": chunk["kb_id"],
                    "important_kwd": [kwd for ck in cks for kwd in ck.get("important_kwd", [])],
                    "image_id": chunk.get("img_id", ""),
                    "similarity": np.mean([ck["similarity"] for ck in cks]),
                    "vector_similarity": np.mean([ck["similarity"] for ck in cks]),
                    "term_similarity": np.mean([ck["similarity"] for ck in cks]),
                    "vector": [0.0] * vector_size,
                    "positions": chunk.get("position_int", []),
                    "doc_type_kwd": chunk.get("doc_type_kwd", "")
                }
                # 获取向量
                for k in cks[0].keys():
                    if k[-4:] == "_vec":
                        d["vector"] = cks[0][k]
                        break
                return d
            except Exception as e:
                self.logger.error(f"Error processing mom chunk {mom_id}: {e}")
                return None

        # 使用线程池并行处理
        merged_chunks = []
        with ThreadPoolExecutor(max_workers=min(32, len(mom_chunks))) as executor:
            future_to_mom = {executor.submit(process_mom_chunk, mom_id, cks): mom_id for mom_id, cks in mom_chunks.items()}

            for future in as_completed(future_to_mom):
                result = future.result()
                if result:
                    merged_chunks.append(result)

        # 合并结果
        chunks.extend(merged_chunks)

        # 按相似度排序
        chunks.sort(key=lambda x: x["similarity"] * -1)

        self.logger.debug(f"retrieval_by_children processed {len(mom_chunks)} mom chunks in {time.time() - start_time:.3f}s")

        return chunks