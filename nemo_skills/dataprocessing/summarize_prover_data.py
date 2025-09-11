import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
import logging
from nemo_skills.utils import (
    get_logger_name,
)
import random
from transformers import AutoTokenizer


LOG = logging.getLogger(get_logger_name(__file__))

def sampling_func(x):
    return int(32*(128+4-x)/(128+4-32))
def temp_func(x):
    if len(x) <= 32:
        return x
    else:
        sample_numble=sampling_func(len(x))
        random.shuffle(x)
        return x[:sample_numble]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    args = parser.parse_args()
    
    data=[]
    base_dir = args.data_path
    total_tiles = 0
    for file in tqdm(os.listdir(base_dir)):
        if file.endswith('.jsonl'):
            with open(os.path.join(base_dir,file),'r') as f:
                for line in f:
                    d=json.loads(line)
                    if d['generation']['success']:
                        if 'promt_turn_list_list' in d['generation']['results_dict_list']:
                            data.append({'problem':d['problem'],'promt_turn_list_list':d['generation']['results_dict_list']['promt_turn_list_list'],'prompt_turn_list':d['generation']['results_dict_list']['prompt_turn_list'],'full_prompt_turn_list':d['generation']['results_dict_list']['full_prompt_turn_list']})
                        else:
                            data.append({'problem':d['problem'],'prompt_turn_list':d['generation']['results_dict_list']['prompt_turn_list']})
            total_tiles += 1
    print(f"Total tiles: {total_tiles}")
    #save as jsonl
    print(f"Saving to {args.output_path}")
    with open(args.output_path+'_raw.jsonl', 'w') as f:
        for d in tqdm(data):
            f.write(json.dumps(d) + '\n')
    
    #postprocessing
    df=pd.DataFrame(data)
    df_grouped=df['promt_turn_list_list'].groupby(df['problem']).apply(list)
    df_filtered=df_grouped.apply(temp_func)   #filter out easy problems with high pass rate
    df_filtered_explode=df_filtered.explode().dropna().to_frame()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dict_list=[]
    #iterate rows as dict 
    df.reset_index(drop=True,inplace=True)
    for i in range(len(df_filtered_explode)):
        d=df_filtered_explode.iloc[i]
        for prompt_turn_list in d['promt_turn_list_list']:
            for dd in prompt_turn_list:
                dd['content']=dd['content'].replace('<|im_start|>assistant','').replace('<|im_start|>','')
            splits=tokenizer.apply_chat_template(prompt_turn_list,tokenize=False).split('<|im_end|>\n<|im_start|>assistant\n')
            inputs='<|im_end|>\n<|im_start|>assistant\n'.join(splits[:-1])+'<|im_end|>\n<|im_start|>assistant\n'
            outputs=splits[-1].replace('<|im_start|>assistant','').replace('<|im_start|>','')
            dict_list.append({'input':inputs,'output':outputs})
    
    with open(args.output_path+'_sft.jsonl','w') as f:
        for d in dict_list:
            f.write(json.dumps(d)+'\n')
