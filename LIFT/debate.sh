STEP=200
for i in {1..15}
do
    # echo "checkpoint-$((STEP * i))"
    python processing_test_debate.py -c $((STEP * i)) -c1 1 -c2 2 -r 2
    # python processing_test_debate.py -c $((STEP * i)) -c1 2 -c2 1 -r 2
    # python debate.py -c /workspace/output_qwen_4_1/checkpoint-$((STEP * i)) -o ./results/output_qwen_4_1/checkpoint-$((STEP * i)) -b 8 -r 2
    # python debate.py -c /workspace/output_qwen_4_2/checkpoint-$((STEP * i)) -o ./results/output_qwen_4_2/checkpoint-$((STEP * i)) -b 8 -r 2
    # python compute_metric_pollen.py -c1 ./results/output_qwen_4_1/checkpoint-$((STEP * i))/result_round_1.json
    # cp /workspace/output_qwen_4_2/checkpoint-$((STEP * i))/result.json ./results/output_qwen_4_2/checkpoint-$((STEP * i))/result.json
done

