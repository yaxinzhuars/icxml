from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import MonoT5
from beir.reranking import Rerank
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from time import time

from beir.retrieval.search.dense import PQFaissSearch, HNSWFaissSearch, FlatIPFaissSearch, HNSWSQFaissSearch 

import pathlib, os
import logging
import json
import random

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=100000)
parser.add_argument('--self_rank', type=bool, default=False)

args = parser.parse_args()

#### Provide the data path where trec-covid has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) trec-covid/corpus.jsonl  (format: jsonlines)
# (2) trec-covid/queries.jsonl (format: jsonlines)
# (3) trec-covid/qrels/test.tsv (format: tsv ("\t"))

# corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
# print(len(corpus))
# print(len(queries))
# # print(corpus[0])
# print(queries[0])

#########################################
#### (1) RETRIEVE Top-100 docs using BM25
#########################################

#### Provide parameters for Elasticsearch
# hostname = "localhost" #localhost
# index_name = "test" # trec-covid
# initialize = True # False

# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
# retriever = EvaluateRetrieval(model)

# #### Retrieve dense results (format of results is identical to qrels)
# results = retriever.retrieve(corpus, queries)

##############################################
#### (2) RERANK Top-100 docs using MonoT5 ####
##############################################

#### Reranking using MonoT5 model #####
# Document Ranking with a Pretrained Sequence-to-Sequence Model 
# https://aclanthology.org/2020.findings-emnlp.63/

#### Check below for reference parameters for different MonoT5 models 
#### Two tokens: token_false, token_true
# 1. 'castorini/monot5-base-msmarco':             ['▁false', '▁true']
# 2. 'castorini/monot5-base-msmarco-10k':         ['▁false', '▁true']
# 3. 'castorini/monot5-large-msmarco':            ['▁false', '▁true']
# 4. 'castorini/monot5-large-msmarco-10k':        ['▁false', '▁true']
# 5. 'castorini/monot5-base-med-msmarco':         ['▁false', '▁true']
# 6. 'castorini/monot5-3b-med-msmarco':           ['▁false', '▁true']
# 7. 'unicamp-dl/mt5-base-en-msmarco':            ['▁no'   , '▁yes']
# 8. 'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['▁não'  , '▁sim']
# 9. 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['▁não'  , '▁sim']
# 10.'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2':['▁não'  , '▁sim']
# 11.'unicamp-dl/mt5-base-en-pt-msmarco-v2':      ['▁no'   , '▁yes']
# 12.'unicamp-dl/mt5-base-mmarco-v2':             ['▁no'   , '▁yes']
# 13.'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['▁no'   , '▁yes']
# 14.'unicamp-dl/mt5-base-mmarco-v1':             ['▁no'   , '▁yes']
# 15.'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['▁não'  , '▁sim']
# 16.'unicamp-dl/ptt5-base-pt-msmarco-100k-v1':   ['▁não'  , '▁sim']
# 17.'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não'  , '▁sim']

def create_batches_dict(dictionary, batch_size):
    batches = []
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i+batch_size]
        batch_values = [dictionary[key] for key in batch_keys]
        batch_dict = dict(zip(batch_keys, batch_values))
        batches.append(batch_dict)
    
    return batches

if args.dataset == "amazon":
    dataset = "LF-Amazon-131K"
if args.dataset == "wiki":
    dataset = "LF-WikiSeeAlso-320K"
if args.dataset == "eurlex":
    dataset = "EURLex-4.3K"
lbl, text = {}, []
with open('../xml/' + dataset + '/lbl.json', encoding='latin-1') as f:
    for i, line in enumerate(f.readlines()):
        data = json.loads(line)
        if args.dataset == "wiki":
            data_uid = str(i)
        if args.dataset in ['amazon', 'eurlex']:
            data_uid = data['uid']
        lbl[data_uid] = {'text': data['title']}
with open('../xml/' + dataset + '/tst.json') as f:
    for line in f.readlines():
        text.append(json.loads(line))


guess = []
with open(args.input_file) as f:
    for i, line in enumerate(f.readlines()):
        # if i < 60000:
        #     continue
        guess.append(json.loads(line))


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# dataset = "trec-covid"

# #### Download nfcorpus.zip dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

corpus = lbl
queries = {i: line['title'] + line['content'] for i, line in enumerate(text[:10000])}

# model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=256, corpus_chunk_size=512*9999)
# retriever = EvaluateRetrieval(model, score_function="dot")
model = models.SentenceBERT("msmarco-distilbert-base-tas-b")
faiss_search = HNSWFaissSearch(model, 
                               batch_size=128, 
                               hnsw_store_n=512, 
                               hnsw_ef_search=128,
                               hnsw_ef_construction=200)

if args.dataset == "amazon":
    prefix = "amazon_lbl_index"       # (default value)
if args.dataset == "wiki":
    prefix = "wiki_lbl_index_num"         # (default value)
if args.dataset == "eurlex":
    prefix = "eurlex_lbl_index"
ext = "hnswpq"              # or "pq", "hnsw", "hnsw-sq"
# input_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "/zdata/users/yazhu/chatgpt/preprocessing/faiss-index")
input_dir = "/zdata/users/yazhu/chatgpt/preprocessing/faiss-index"

if os.path.exists(os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))):
    faiss_search.load(input_dir=input_dir, prefix=prefix, ext=ext)
retriever = EvaluateRetrieval(faiss_search, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
# start_time = time()
# results = retriever.retrieve(corpus, queries)
# end_time = time()
# print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

# out = {}
# top_k = 100
# for query_id, ranking_scores in results.items():
#     i = query_id
#     if i not in out.keys():
#         out[i] = []
#     scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
#     for rank in range(top_k):
#         doc_id = scores_sorted[rank][0]
#         hit = corpus[doc_id]['text']
#         if hit not in out[i]:
#             out[i].append(hit)

# with open('../preprocessing/guess_10000_all_beir.jsonl', "w") as outfile:
#     for i in range(len(text[:10000])):
#         outfile.write(
#             json.dumps(
#                 {
#                     "id": text[i]['uid'],
#                     "output": out[i]
#                 }
#             ) + "\n"
#         )



#### generate -> label
print(len(lbl))

open(args.save_to, "w")

queries = {}
top_k = 10
for i, guess_line in enumerate(guess):
    for j, item in enumerate(guess_line['output']):
        queries[str(i) + ':' + str(j)] = item

batches = create_batches_dict(queries, args.batch_size)

out = {}
tmp = {}
tmp_q = {}
for queries_batch in batches:
    results = retriever.retrieve(corpus, queries_batch)
    for query_id, ranking_scores in results.items():
        i, j = [int(x) for x in query_id.split(':')]
        if (i + args.start) not in out.keys():
            out[i + args.start] = []
            tmp_q[i + args.start] = {}
        if j not in tmp_q[i + args.start].keys():
            tmp_q[i + args.start][j] = []
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            hit = corpus[doc_id]['text']
            if hit != text[i + args.start]['title']:
                tmp_q[i + args.start][j].append(hit)
            if hit not in out[i + args.start] and hit != text[i + args.start]['title']:
                out[i + args.start].append(hit)

print(len(tmp_q))

if args.self_rank:
    # for i in range(len(tmp)): 
    #     o = []
    #     for j in sorted(tmp[i].keys()):
    #         # print(i, j)
    #         # print(len(tmp[i][j]))
    #         o.extend(tmp[i][j])
    #     assert len(o) == len(out[i])
    #     tmp_q[i] = o

    for i in sorted(tmp_q.keys()): 
        o = []
        empty = {}
        # empty = {j: True for j in sorted[tmp_q][i].keys()}
        for j in sorted(tmp_q[i].keys()):
            empty[j] = True
        while any(empty.values()):
            for j in sorted(tmp_q[i].keys()):
                # print(i, j)
                # print(len(tmp[i][j]))
                if len(tmp_q[i][j]) > 0:
                    x = tmp_q[i][j].pop(0)
                    if x not in o:
                        o.append(x)
                else:
                    empty[j] = False
        # assert len(o) == len(out[i])
        tmp_q[i] = o
    with open(args.save_to, "w") as outfile:
        for i in range(len(guess)):
            outfile.write(
                json.dumps(
                    {
                        "id": text[i + args.start]['uid'],
                        "output": tmp_q[i + args.start]
                    }
            ) + "\n"
        )
else:

    with open(args.save_to, "w") as outfile:
        for i in range(len(guess)):
            outfile.write(
                json.dumps(
                    {
                        "id": text[i + args.start]['uid'],
                        "output": out[i + args.start]
                    }
            ) + "\n"
        )

