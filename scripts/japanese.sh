source venv/bin/activate
python "mamba_generate_custom_model.py" \
--prompt "東京とは、" \
--topp 0.9 \
--temperature 1 \
--max-length 200 \
--tokenizer "loiccabannes/MambaSan-370m-instruct" \
--model-name "loiccabannes/MambaSan-370m-instruct" \
--device "mps"
# --device "cpu"