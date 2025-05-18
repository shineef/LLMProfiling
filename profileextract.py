from ollama import chat
from ollama import ChatResponse
import re

import csv

import torch


def token_dropping_llm(item, model, tokenizer, device):
    title = item[1]

    inputs = tokenizer(item[2], max_length=1024, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, do_sample=False)
    abstract = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    new_item = [item[0], title, abstract]
    return new_item

def LLMExtracter(item):
    description = item[2]

    response: ChatResponse = chat(model='qwq', messages=[
      {
        'role': 'user',
        'content': f'Your task is to summarize the abstract of the paper while retaining as much useful content as possible: \n {description} \n Output the summarized content ONLY.',
      },
    ])

    content = response['message']['content']


    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    return [item[0], item[1], cleaned_content]

def read_tsv(file_path):
    with open(file_path, 'rb') as file: 
        content = file.read().decode('utf-8', 'replace').replace('\0', '')
    data = [row for row in csv.reader(content.splitlines(), delimiter='\t')]
    return data

def main():
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        print(f"current gpu:{gpu_name} (Index: {gpu_index})")
    else:
        print("CPU")


    with open("train_dataset_origin_clean.tsv", 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        # header = next(reader) 
        data = list(reader)

    new_data = []

    for item in data:
        new_item = LLMExtracter(item)
        new_data.append(new_item)

    with open("qwq_titleabs.tsv", 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(new_data)
        

if __name__ == "__main__":
    main()