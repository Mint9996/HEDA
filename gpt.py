
import argparse
import openai
import json
import re
import time
from tqdm import tqdm





openai.api_key = ''
openai.api_base = ''

def get_res_batch(input,label):
    message = [
        {"role": "user", "content":input + '\n' + "The answer is '" + label + "'. Let's think step by step."}
        # {"role": "user", "content":input + '\n' + "Why the answer is '" + label + "'?"} 
    ]

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0,
                max_tokens=128
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(1)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(1)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(1)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(1)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(1)
    
    

    return res['choices'][0]['message']['content']

if __name__ == '__main__':
    file = open('data/mnli/train2.src','r')
    src=file.readlines()
    file.close()
    label_f = open('data/mnli/train2.tgt','r')
    labels=label_f.readlines()
    label_f.close()   


    tgt_file=open('data/mnli/train2_exp2.tgt','w')
    for i in tqdm(len(src)):
        label = labels[i]
        result=get_res_batch((src[i].replace("\n","")),(label.replace("\n","")))
        tgt_file.write(result.replace("\n","")+"\n")