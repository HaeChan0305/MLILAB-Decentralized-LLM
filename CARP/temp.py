import json

dataset_name = "sst2"

choices = {
    "agnews" : ["Sports", "World", "Science/Technology", "Business"],
    "mr" : ["Positive", "Negative"],
    "r8" : ["Grain", "Earnings and Earnings Forecasts", "Interest Rates", "Money/Foreign Exchange", "Acquisitions", "Crude Oil", "Shipping", "Trade"],
    "sst2" : ["Positive", "Negative"],
}
 

with open(f"output_5_0_1/checkpoint-0/{dataset_name}_result.json") as file:
    data = json.load(file)
    
invalid_6_list = [example['prediction'] for example in data if example['prediction'] not in choices[dataset_name]]
print(len(invalid_6_list))

#for i in invalid_6_list:
#     print(i)
