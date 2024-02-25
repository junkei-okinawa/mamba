source ~/miniforge3/etc/profile.d/conda.sh
conda activate mamba_cpu
python "mamba_generate_custom_model.py" \
--prompt "Tell me 5 sites to visit in Spain" \
--topp 0.9 \
--temperature 1 \
--max-length 700 \
--tokenizer "clibrain/mamba-2.8b-instruct-openhermes" \
--model-name "clibrain/mamba-2.8b-instruct-openhermes" \
--device "mps"
# --device "cpu"