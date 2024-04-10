# ko_llm_eval
-------------

한국어 llm 성능 평가 관련 repo. 

## Korean MT-Bench
- 한국어 MT-Bench 벤치마크 데이터셋으로 LLM 성능평가 진행
- FastChat/fastchat/llm_judge 디렉토리 내 소스코드로 성능평가 진행가능

- MT-Bench score를 계산하기 위해서는 다음 3가지 step을 거쳐야 한다.

> 1. 성능 평가 대상 LLM으로 벤치마크 데이터셋의 question에 대해 answer를 생성한다.
> 2. GPT-4를 이용해 LLM이 생성한 answer를 평가한다.
> 3. 각 생성문에 대한 score를 종합하여 최종 MT-bench score를 계산한다.

'''
cd fastchat/llm_judge

# Step 1. Generate model answers to MT-bench questions
# This will generate `data/korean_mt_bench/model_answer/42dot_LLM-SFT-1.3B.jsonl`
python gen_model_answer.py --bench-name korean_mt_bench --model-path 42dot/42dot_LLM-SFT-1.3B --model-id 42dot_LLM-SFT-1.3B

# Step 2. Generate GPT-4 judgments
# This will generate `data/korean_mt_bench/model_judgment/gpt-4_single.jsonl`
OPENAI_API_KEY=<API-KEY> python gen_judgment.py --bench-name korean_mt_bench --model-list 42dot_LLM-SFT-1.3B --judge-file data/judge_ko_prompts.jsonl


# Step 3. Show MT-bench scores
python show_result.py --bench-name korean_mt_bench --input-file <step2.에서 생성한 answer jsonl 파일>
'''
