source venv/bin/activate
python "mamba_generate_custom_model.py" \
--prompt "Write a python script to remove .tmp files" \
--topp 0.9 \
--temperature 1 \
--max-length 300 \
--tokenizer "mrm8488/mamba-coder" \
--model-name "mrm8488/mamba-coder" \
--device "mps"
# --device "cpu"