read -p "please input your dataset name:" name
read -p "please input your cuda number:" cu
python model/main.py \
   --gpu $cu \
   --n-epochs 30 \
   --dim 128 \
   --bsize 32 \
   --query_maxlen 256 \
   --doc_maxlen 256 \
   --special-tokens "QARCL" \
    --data-dpath "./data/"$name \
   --triples "training_dataset_RN_commits.csv" \
   --config "BERTOverflow" &&
python model/indexer.py \
   --gpu $cu \
   --bsize 32 \
   --dim 128 \
   --query_maxlen 256 \
   --doc_maxlen 256 \
   --checkpoint "./data/"$name"/model_SemanticCodebert_"$name"_RN_bertoverflow_QARCL_q256_d256_dim128_cosine_commits" \
   --data-dpath "./data/"$name \
   --chunksize 6.0 &&
python model/faiss_indexer.py \
   --gpu $cu \
   --data-dpath "./data/"$name \
   --index-name "INDEX_SemanticCodebert_RN_bertoverflow_QARCL_q256_d256_dim128_cosine_q256_d256_dim128_commits_token" \
   --partitions 256 &&
python model/ranking.py \
   --gpu $cu \
   --dim 128 \
   --query-maxlen 256 \
   --doc-maxlen 256 \
   --bsize 32 \
   --data-dpath "./data/"$name \
   --checkpoint "./data/"$name"/model_SemanticCodebert_"$name"_RN_bertoverflow_QARCL_q256_d256_dim128_cosine_commits" \
   --index_name "INDEX_SemanticCodebert_RN_bertoverflow_QARCL_q256_d256_dim128_cosine_q256_d256_dim128_commits_token" \
   --faiss_name "ivfpq.256.faiss" \
   --faiss_depth 1024 \
   --nprobe 100 &&
cd "results" &&
python accuracy.py \
   --n-epochs 30 \
   --bsize 32 \
   --k 32 \
   --dataset $name