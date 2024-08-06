for data in agnews mr r8 sst2
do
    # echo $data
    # python test.py -d $data -o ./output_5_0_1/checkpoint-0/${data}_result_2.json -b 31 -e False
    python compute_metric.py -d $data -r ./output_5_0_1/checkpoint-0/${data}_result.json -e False
done

# python test.py -d r8 -o ./output_5_0_1/checkpoint-0/r8_result_2.json -b 17 -e False

# python compute_metric.py -d agnews -r ./output_5_0_2/checkpoint-0/agnews_result.json -e True