source venv/bin/activate
python "mamba_generate_custom_model.py" \
--prompt "こんにちは。お元気ですか？" \
--topp 0.9 \
--temperature 1 \
--max-length 100 \
--tokenizer "EleutherAI/gpt-neox-20b" \
--model-name "kotoba-tech/kotomamba-2.8B-CL-v1.0" \
--device "mps"
# --device "cpu"