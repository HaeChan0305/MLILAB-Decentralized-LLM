data=mr
client=2

python compute_metric.py -d $data -r ./output_5_2_${client}/${data}/${data}_result.json
for round in {1..8}
do
    python compute_metric.py -d $data -r ./output_5_2_${client}/${data}/${data}_result_round_${round}.json
done

# EXP 5_1_1 / EXP 5_1_2
# data=r8
# STEP=85
# client=2
# for i in {1..10}
# do
#     python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result.json
# done
