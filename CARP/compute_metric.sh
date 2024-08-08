# data=mr
# STEP=283
# for i in {1..10}
# do
#     python compute_metric.py -d $data -r ./output_5_0_2/${data}/checkpoint-$((STEP * i))/${data}_result.json -e False
# done

data=r8
STEP=171
for i in {1..10}
do
    python compute_metric.py -d $data -r ./output_5_0_2/${data}/checkpoint-$((STEP * i))/${data}_result.json -e False
done

# data=mr
# STEP=283
# for i in {1..3}
# do
#     python compute_metric.py -d $data -r ./output_test/${data}/checkpoint-$((STEP * i))/${data}_result.json -e True
# done