import os
import json
from typing import Dict, Any, List

def read_data(file_path: str, split: str = 'dev') -> List[Dict[str, Any]]:
    """
    Read data from either dev.json or test.json file.

    Input:
        file_path: str, the path to the directory containing JSON files
        split: str, either 'dev' or 'test'
    Output:
        List[Dict[str, Any]]: a list of dictionaries with the data from the file
    """
    data = []
    
    # 确保split参数正确
    if split not in ['dev', 'test']:
        raise ValueError("split must be either 'dev' or 'test'")
    
    # 读取指定的json文件
    filename = f'{split}.json'
    full_path = os.path.join(file_path, filename)
    
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as file:
            batch_data = json.load(file)
            for item in batch_data:
                text = item['text']
                images = item['images']
                if all(os.path.exists(image) for image in images):
                    data.append({
                        'text': text,
                        'images': images,
                        'label': item['label']
                    })
    else:
        raise FileNotFoundError(f"{filename} not found in {file_path}")
    
    return data

dataset=read_data("/projects/vig/hzy/XFacta")
print(dataset[0:5])

