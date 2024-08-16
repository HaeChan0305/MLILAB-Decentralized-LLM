data=mr
STEP=141
client=2
round=2
for i in {1..10}
do
    python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result_round_${round}.json
done

# data=r8
# STEP=85
# client=2
# for i in {1..10}
# do
#     python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result.json
# done
