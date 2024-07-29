import json

STEP = 750
PATH = "gsm8k/output_qwen_1_1/checkpoint-{0}/result.json"
CHECKPOINTS = [i * STEP for i in range(2)]

for checkpoint in CHECKPOINTS:
    path = PATH.format(checkpoint)    
    
    with open(path, "r") as file:
        data = json.load(file)


    total_len = 0
    max_len = 0
    cnt_1024 = 0
    cnt_invalid = 0
    for i, d in enumerate(data):
        if i == 62:
            if checkpoint == 0:
                print("========= Question =========")
                print(d['question'])
                
                print("========= Answer =========")
                print(d['answer'])
                
                print("========= Prediction : checkpoint-0  =========")
                print(d['prediction'])
            
            else:
                print("========= Prediction : checkpoint-750  =========")
                print(d['prediction'])
        
        # total_len += len(d['prediction'])
        # if max_len < len(d['prediction']):
        #     max_len = len(d['prediction'])
        # if 1024 < len(d['prediction']):
        #     cnt_1024 += 1
            

    # print(f"\n========= {path} =========")
    # print(cnt_invalid)
    # print("length avg : ", total_len/len(data))
    # print("length max : ", max_len)
    # print("num > 1024 : ", cnt_1024)
    print()