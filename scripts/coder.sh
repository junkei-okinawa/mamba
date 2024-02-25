source ~/miniforge3/etc/profile.d/conda.sh
conda activate mamba_cpu
python "mamba_generate_custom_model.py" \
--prompt "Write a rust script to remove .tmp files" \
--topp 0.9 \
--temperature 1 \
--max-length 300 \
--tokenizer "mrm8488/mamba-coder" \
--model-name "mrm8488/mamba-coder" \
--device "mps"
# --device "cpu"