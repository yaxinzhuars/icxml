python src/group.py \
--prompt_type example \
--dataset  \
--lbl_file  \
--test_file \
--hint_file (ignore when not using hint) \
--save_to file1 \
--start \
--end \

python src/read_example.py \
--input_file file1 \
--save_to file2 \


python group.py \
--prompt_type group \
--dataset  \
--lbl_file  \
--test_file  \
--hint_file  \
--example_file file2 \
--save_to file3 \
--start \
--end \

python src/clean_data.py \
--train_mode full \
--dataset  \
--input_file file3 \
--save_to file4 \

python src/beirtest.py \
--dataset amazon \
--input_file file4 \
--save_to file5 \
--start  \
--end  \
--self_rank True


python src/rerank.py \
--prompt_type select \
--dataset  \
--save_to file6 \
--input_file file5 \
--start  \
--end  \

python src/read_rank.py \
--dataset amazon \
--input_file file6 \
--gt_file file5 \
--save_to file7 \

python eval.py --gold gold file --guess file7 --k 1,3,5,10