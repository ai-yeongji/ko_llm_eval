# pip install google.generativeai

CUDA_VISIBLE_DEVICES=0 poetry run python scripts/evaluate_llm.py -cn config.yaml \
    model.pretrained_model_name_or_path=/data/public/EXAONE_IT_1.7/chat_exaone_mt_240118_e3 \
  tokenizer.pretrained_model_name_or_path=/data/public/EXAONE_IT_1.7/chat_exaone_mt_240118_e3 \
  dataset_dir=dataset/1.2.0/evaluation/test

  # CUDA_VISIBLE_DEVICES=0 poetry run python scripts/evaluate_llm.py -cn config.yaml model.pretrained_model_name_or_path=beomi/llama-2-ko-7b tokenizer.pretrained_model_name_or_path=beomi/llama-2-ko-7b dataset_dir=dataset/1.2.0/evaluation/test
