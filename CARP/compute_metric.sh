data=mr
STEP=141
client=2
round=1
for i in {1..10}
do
    python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result_round_${round}.json >> ./output_5_1_${client}/${data}/result_round_${round}.txt
done

# data=r8
# STEP=85
# client=2
# for i in {1..10}
# do
#     python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result.json
# done
