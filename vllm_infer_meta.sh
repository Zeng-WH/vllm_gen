
python vllm_infer_wandb_hf.py \
    --input_data /share/project/weihao/save_dir/checkpoints/sft_metamath_trn_5e_5_3epoch/ckpt18513_hf/syn_120k_data_4infer_part_4.json \
    --model_dir /share/project/weihao/save_dir/checkpoints/sft_metamath_trn_5e_5_3epoch/ckpt18513_hf \
    --sample_num 64 \
    --output_file /share/project/weihao/save_dir/checkpoints/sft_metamath_trn_5e_5_3epoch/ckpt18513_hf/syn_120k_data_4infer_part_4_sample64.json \
    --tensor_parallel_size 8 \
    --temperature 1.2 \
    --top_k 40 \
    --max_tokens 768 \
    --repo_id AndrewZeng/math_scaling \
    --hf_token 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 \
    --wandb_project vllm_gen \