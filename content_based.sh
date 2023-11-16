python src/demo_gen.py \
--prompt_type example \
--dataset \
--save_to file1 \
--start  \
--end \

python src/read_example.py \
--input_file file1 \
--save_to file2 \

python src/beir_example.py \
--dataset  \
--input_file file2 \
--save_to file3 \
--start 0 \

python src/inference.py \
--prompt_type example \
--dataset \
--lbl_file  \
--test_file  \
--pseudo_file  \
--shots_file file3 \
--save_to file4 \
--start \
--end \

python src/clean_data.py \
--train_mode full \
--dataset  \
--input_file file4 \
--save_to file5 \

python src/beirtest.py \
--dataset amazon \
--input_file file5 \
--save_to file6 \
--start  \
--end  \
--self_rank True


python src/rerank.py \
--prompt_type select \
--dataset  \
--save_to file7 \
--input_file file6 \
--start  \
--end  \

python src/read_rank.py \
--dataset amazon \
--input_file file7 \
--gt_file file6 \
--save_to file8 \

python eval.py --gold gold file --guess file8 --k 1,3,5,10