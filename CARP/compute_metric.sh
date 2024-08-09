# data=mr
# STEP=283
# for i in {0..10}
# do
#     python compute_metric.py -d $data -r ./output_5_0_3/${data}/checkpoint-$((STEP * i))/${data}_result.json
# done

data=r8
STEP=171
for i in {0..10}
do
    python compute_metric.py -d $data -r ./output_5_0_3/${data}/checkpoint-$((STEP * i))/${data}_result.json
done
