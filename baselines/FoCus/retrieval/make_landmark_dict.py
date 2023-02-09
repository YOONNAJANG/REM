#
# json형태가
#     {'landmark이름 or link':[
#             {"dpr":['문장기준으로 split되어서 DPR tokenizer로 id로 바뀐 문장들']},
#             {"bart":['문장기준으로 split되어서 BART tokenizer로 id로 바뀐 문장들']},
#             {"gpt":['문장기준으로 split되어서 GPT tokenizer로 id로 바뀐 문장들']}]
#     }
# 실제 코드에서는
# 1) history(question)을 bart(gpt) tokenizer로 디코딩하고 DPRTokenizer로 인코딩하고
# 2) link보고 id로 변환되어 있는 문장들(knowledge) 가져와서
# 3) 1),2)번 가지고 DPR모델 입력을 만들어줌 : [CLS] <question token ids> [SEP] <texts ids> [SEP]
# 4) 모델에 넣어서 relevance logits 구함
#
from transformers import DPRReader, DPRReaderTokenizer
from transformers import BartTokenizer, GPT2Tokenizer, LEDTokenizer, T5Tokenizer
import json
dpr_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
led_tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
data_name = ["/home/mnt/yoonna/focus_modeling/our_data/train_ours.json","/home/mnt/yoonna/focus_modeling/our_data/valid_ours.json","/home/mnt/yoonna/focus_modeling/our_data/test_ours.json"]
dataset_enc = dict()
for file in data_name:
    with open(file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        for dialogue in dataset["data"]: #len 8
            temp = {}
            total_sen_k = []
            knowledge = dialogue["knowledge"]
            landmark = dialogue["landmark_link"]
            print(landmark)
            for k in knowledge:
                sen_k = k.split(". ")
                total_sen_k.extend(sen_k)
            print("len(total_sen_k):   ",len(total_sen_k))
            enc_dpr = dpr_tokenizer(total_sen_k)["input_ids"]
            print("len(enc_dpr):    ", len(enc_dpr))
            enc_bart = bart_tokenizer(total_sen_k)["input_ids"]
            print("len(enc_bart):    ", len(enc_bart))
            enc_gpt2 = gpt_tokenizer(total_sen_k)["input_ids"]
            print("len(enc_gpt2):    ", len(enc_gpt2))
            enc_led = led_tokenizer(total_sen_k)["input_ids"]
            print("len(enc_led):    ", len(enc_led))
            enc_t5 = t5_tokenizer(total_sen_k)["input_ids"]
            print("len(enc_t5):    ", len(enc_t5))

            temp["dpr"] = enc_dpr
            temp["bart"] = enc_bart
            temp["gpt2"] = enc_gpt2
            temp["t5"] = enc_t5
            temp["led"] = enc_led
            dataset_enc[landmark] = temp

with open("/home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json", "w") as json_file:
    json.dump(dataset_enc, json_file)