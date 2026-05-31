from retrieval.google.Image2text import get_text_using_image
from retrieval.google.Caption2image import get_image_using_caption
from retrieval.google.Caption2text import get_text_using_caption
from retrieval.google.question_gen import run_generation_pipeline
from retrieval.duckduckgo.evidence_search import get_evidence_news, get_evidence_text, get_evidence_images
import os
import json
import tqdm
from dotenv import load_dotenv
import re
import requests
from PIL import Image
from io import BytesIO
load_dotenv()

"""
1. This file is primarily used for retrieving evidence data from various sources. 
It utilizes multiple external modules and APIs to achieve this functionality.
- `get_text_using_image`: Extracts text from images.
- `get_image_using_caption`: Generates images based on captions.
- `get_text_using_caption`: Generates text based on captions.
- `run_generation_pipeline`: Question generation pipeline.
- `get_evidence_news`, `get_evidence_text`, `get_evidence_images`: Retrieve news, text, and image evidence from DuckDuckGo.

2. The file defines a function `load_evidence` for loading evidence data. This function accepts three parameters:
1. `data`: A list containing sample data.
2. `include_evidences`: A list of integers specifying the types of evidence to include.

3. The `load_evidence` function calls different APIs to obtain evidence based on the contents of `include_evidences` and stores it in the sample data.
`include_evidences` is a list of integers specifying the types of evidence to include.
- 1: Extract text from images.
- 2: Generate images based on captions.
- 3: Generate text based on captions.
- 4: Retrieve news from DuckDuckGo.
- 5: Retrieve text from DuckDuckGo.
- 6: Retrieve images from DuckDuckGo.
- 7: Generate questions and search for text evidence based on the questions.
- 8: Generate questions and search for image evidence based on the questions.
"""

def download_image(image_url, folder_path, image_name):
    try:
        # Extract only the image URL, removing the unnecessary text before it
        clean_url = image_url.split("URL:")[0].strip().replace("[Image Evidence]:", "").strip()
        response = requests.get(clean_url)
        response.raise_for_status()  # Check for request errors
        image = Image.open(BytesIO(response.content))
        
        # Save the image locally with the given name
        image_path = os.path.join(folder_path, f"{image_name}.jpeg")
        image.save(image_path, "JPEG")
        return image_path  # Return the local path for later use

    except Exception as e:
        return None

untrusted_sources = {
    "www.facebook.com", "m.facebook.com", "www.reddit.com", "www.weibo.com", "twitter.com",
    "www.tiktok.com", "www.douyin.com", "www.instagram.com", "www.pinterest.com", "www.taobao.com",
    "www.jd.com", "www.amazon.com", "www.ebay.com", "www.imdb.com", "www.douban.com", 
    "steamcommunity.com", "m.ixigua.com", "www.bilibili.com", "www.netflix.com"
}

untrusted_sources_de = {
    "twitter.com", "pbs.twimg.com", "www.twitter.com"
}

def extract_domain(url):
    if not url:
        return None
    match = re.search(r'https?://([^/]+)', url)
    if match:
        return match.group(1)
    return None

def is_untrusted_evidence(evidence, domain_blocklist):
    if isinstance(evidence, dict):
        for key in ['img_link', 'domain', 'url']:
            if key in evidence:
                domain = extract_domain(evidence[key]) if key != 'domain' else evidence[key]
                if domain and domain in domain_blocklist:
                    return True
        for field in ['title', 'snippet', 'text', 'body']:
            if field in evidence and isinstance(evidence[field], str):
                for domain in domain_blocklist:
                    if domain in evidence[field]:
                        return True
    if isinstance(evidence, str):
        for domain in domain_blocklist:
            if domain in evidence:
                return True
    return False


def load_evidence(data, include_evidences, filter_untrusted=True, top_k=5, dataset_type='dev',evidence_cache=False):
    domain_blocklist = untrusted_sources if filter_untrusted else untrusted_sources_de
    if not isinstance(include_evidences, list):
        raise ValueError(f"Expected include_evidences to be a list, but got {type(include_evidences)}")

    if not include_evidences:
        for sample in data:
            sample['evidence1'] = ""
            sample['evidence2'] = ""
            sample['evidence3'] = ""
            sample['evidence4'] = ""
            sample['evidence5'] = ""
            sample['evidence6'] = ""
            sample['evidence7'] = ""
            sample['evidence8'] = ""
        return data

    if 1 in include_evidences: 
        save_folder_path = f"./output/image2text/{dataset_type}"
        api_key = os.getenv("GOOGLE_VISION_API_KEY")
        text_search_query_num = 5
        for idx, sample in enumerate(data, start=1): 
            image_paths = sample.get('images', [])
            evidence_list = []
            cache_file_path = os.path.join(save_folder_path, f"{idx}/image_text/annotation.json")
            
            if evidence_cache:
                if os.path.exists(cache_file_path):
                    print(f"Loading cached evidence for sample {idx} from {cache_file_path}")
                    try:
                        with open(cache_file_path, 'r', encoding='utf-8') as cache_file:
                            cached_evidence = json.load(cache_file)
                            if not cached_evidence:
                                print(f"Cache file {cache_file_path} is empty. Proceeding with new retrieval.")
                            else:
                                for ann in cached_evidence:
                                    if isinstance(ann, dict):
                                        evidence_list.append(f"[Title]: {ann.get('title', 'No Title')}\n[Text]: {ann.get('text', 'No Text')}\n[URL]: {ann.get('url', 'No URL')}")
                                    else:
                                        evidence_list.append(str(ann))
                    except Exception as e:
                        print(f"Error loading cached evidence for sample {idx}: {e}")
                        print(f"Proceeding with new evidence retrieval.")
                else:
                    print(f"Cache file {cache_file_path} does not exist, raising error.")
                    raise FileNotFoundError(f"Cache file {cache_file_path} does not exist, unable to load evidence.")
            else:
                for i, image_path in enumerate(image_paths):
                    try:
                        uid = str(idx)
                        annotations = get_text_using_image(
                            image_path=image_path,
                            save_folder_path=save_folder_path,
                            api_key=api_key,
                            text_search_query_num=text_search_query_num,
                            uid=uid
                        )
                        if annotations:
                            annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                            for ann in annotations:
                                if isinstance(ann, dict):
                                    evidence_list.append(f"[Title]: {ann.get('title', 'No Title')}\n[Text]: {ann.get('text', 'No Text')}\n[URL]: {ann.get('url', 'No URL')}")
                                else:
                                    evidence_list.append(str(ann))
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                
                sample['evidence1'] = "\n\n".join(evidence_list) if evidence_list else ""

            sample['evidence1'] = "\n\n".join(evidence_list) if evidence_list else ""


    if 2 in include_evidences: 
        save_folder_path = f"./output/caption2image/{dataset_type}"
        api_key = os.getenv("GOOGLE_VISION_API_KEY")
        cse_id = os.getenv("cse_id")
        image_search_query_num = 1
        base64 = False  
        os.makedirs(save_folder_path, exist_ok=True)

        for idx, sample in enumerate(data, start=1):  
            caption = sample['text'].strip('\'"')  
            evidence_list = []
            uid = str(idx)
            annotation_file = os.path.join(save_folder_path, uid, "caption_image/annotation.json")

            if evidence_cache:
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        annotations = json.load(f)
                    annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                    for ann in annotations:
                        if 'img_link' in ann and 'page_link' in ann and 'domain' in ann and 'snippet' in ann:
                            local_image_path = ann.get('image_path', None)
                            if local_image_path:
                                evidence_list.append(f"[local_image_path]: {local_image_path}")
                            else:
                                print(f"The local image path is missing.")
                    print(f"Use cache files：{annotation_file}")
                except Exception as e:
                    print(f"Failed to read the cache：{annotation_file}，错误：{e}")
            else:
                try:
                    uid = str(idx) 
                    annotations = get_image_using_caption(
                        text_query=caption,
                        save_folder_path=save_folder_path,
                        api_key=api_key,
                        cse_id=cse_id,
                        image_search_query_num=image_search_query_num,
                        uid=uid,
                        base64=base64
                    )
                    if annotations:
                        annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                    for ann in annotations:
                        if 'img_link' in ann and 'page_link' in ann and 'domain' in ann and 'snippet' in ann:
                            local_image_path = ann.get('local_image_path', None)
                            if local_image_path:
                                evidence_list.append(f"[local_image_path]: {local_image_path}")
                            else:
                                print(f"Missing local image path")
                
                except Exception as e:
                    print(f"Error processing caption {caption}: {e}")

            sample['evidence2'] = "\n\n".join(evidence_list[:top_k]) if evidence_list else ""


    if 3 in include_evidences:  
        save_folder_path = f"./output/caption2text/{dataset_type}" 
        api_key = os.getenv("GOOGLE_VISION_API_KEY")
        cse_id = os.getenv("cse_id")
        text_search_query_num = 2
        os.makedirs(save_folder_path, exist_ok=True)

        for idx, sample in enumerate(data, start=1):
            text_query = sample['text'].strip('\'"') 
            evidence_list = []
            uid = str(idx)
            annotation_file = os.path.join(save_folder_path, uid, "caption_text/annotation.json")

            if evidence_cache:
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        annotations = json.load(f)
                    annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                    for ann in annotations:
                        if 'text' in ann:
                            evidence_list.append(ann['text'])
                    print(f"Use the cache file:{annotation_file}")
                except Exception as e:
                    print(f"Failed to read the cache:{annotation_file}，错误：{e}")
            else:
                try:
                    uid = str(idx) 
                    annotations = get_text_using_caption(
                        text_query=text_query,
                        save_folder_path=save_folder_path,
                        api_key=api_key,
                        cse_id=cse_id,
                        text_search_query_num=text_search_query_num,
                        uid=uid,
                        do_domain_filter=False 
                    )
                    if annotations:
                        annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                    for ann in annotations:
                        if 'text' in ann:
                            evidence_list.append(ann['text'])

                    if evidence_list:
                        sample['evidence3'] = "\n\n".join(evidence_list[:top_k])
                    else:
                        sample['evidence3'] = "" 

                except Exception as e:
                    print(f"Error processing text query {text_query}: {e}")
                    sample['evidence3'] = "" 
            sample['evidence3'] = "\n\n".join(evidence_list[:top_k]) if evidence_list else ""

    if 4 in include_evidences: 
        save_folder_path = f"./output/duckduckgo_news/{dataset_type}"
        os.makedirs(save_folder_path, exist_ok=True)

        for idx, sample in enumerate(data, start=1):
            text_query = sample['text'].strip('\'"')
            evidence_list = []
            uid = str(idx)
            annotation_file = os.path.join(save_folder_path, uid, "annotations.json")

            if evidence_cache and os.path.isfile(annotation_file):
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        evidence_dict = json.load(f)
                    if 'Information related to the text' in evidence_dict:
                        for entry in evidence_dict['Information related to the text']:
                            if "Body:" in entry:
                                evidence_part = entry.split("Body:")[1].strip()
                                evidence_list.append(evidence_part)
                    print(f"Use cache files:{annotation_file}")
                except Exception as e:
                    print(f"Failed to read the cache:{annotation_file}，错误：{e}")
            else:
                try:
                    evidence = get_evidence_news(text_query)
                    if isinstance(evidence, str):
                        try:
                            evidence_dict = json.loads(evidence)
                            
                            if 'Information related to the text' in evidence_dict:
                                for entry in evidence_dict['Information related to the text']:
                                    if "Body:" in entry:
                                        evidence_part = entry.split("Body:")[1].strip()
                                        evidence_list.append(f"{evidence_part}")
                                        
                            annotation_dir = os.path.join(save_folder_path, uid)
                            os.makedirs(annotation_dir, exist_ok=True)
                            annotation_path = os.path.join(annotation_dir, "annotations.json")
                            with open(annotation_path, "w", encoding="utf-8") as f:
                                json.dump(evidence_dict, f, indent=2, ensure_ascii=False)

                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON evidence: {e}")

                    if evidence_list:
                        sample['evidence4'] = "\n\n".join(evidence_list[:top_k])
                    else:
                        sample['evidence4'] = ""

                except Exception as e:
                    print(f"Error processing text query {text_query}: {e}")
                    sample['evidence4'] = "" 
            sample['evidence4'] = "\n\n".join(evidence_list[:top_k]) if evidence_list else ""


    if 5 in include_evidences: 
        save_folder_path = f"./output/duckduckgo_text/{dataset_type}"
        os.makedirs(save_folder_path, exist_ok=True)

        for idx, sample in enumerate(data, start=1):
            text_query = sample['text'].strip('\'"')  
            evidence_list = []
            uid = str(idx)
            annotation_file = os.path.join(save_folder_path, uid, "annotations.json")

            if evidence_cache:
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        evidence_dict = json.load(f)
                    if 'Information related to the text' in evidence_dict:
                        for entry in evidence_dict['Information related to the text']:
                            if "Evidence:" in entry:
                                evidence_part = entry.split("Evidence:")[1].strip()
                                evidence_list.append(evidence_part)
                    print(f"Use cache file:{annotation_file}")
                except Exception as e:
                    print(f"Failed to read the cache:{annotation_file}，错误：{e}")
            else:
                try:
                    evidence = get_evidence_text(text_query)
                    print(evidence)
                    if isinstance(evidence, str):
                        try:
                            evidence_dict = json.loads(evidence)
                            
                            if 'Information related to the text' in evidence_dict:
                                for entry in evidence_dict['Information related to the text']:
                                    if "Evidence:" in entry:
                                        evidence_part = entry.split("Evidence:")[1].strip()  
                                        evidence_list.append(f"{evidence_part}")
                            
                            annotation_dir = os.path.join(save_folder_path, uid)
                            os.makedirs(annotation_dir, exist_ok=True)
                            annotation_path = os.path.join(annotation_dir, "annotations.json")
                            with open(annotation_path, "w", encoding="utf-8") as f:
                                json.dump(evidence_dict, f, indent=2, ensure_ascii=False)

                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON evidence: {e}")

                    if evidence_list:
                        sample['evidence5'] = "\n\n".join(evidence_list[:top_k])
                    else:
                        sample['evidence5'] = ""  

                except Exception as e:
                    print(f"Error processing text query {text_query}: {e}")
                    sample['evidence5'] = "" 
            sample['evidence5'] = "\n\n".join(evidence_list[:top_k]) if evidence_list else ""




    if 6 in include_evidences:
        def is_valid_image_file(path):
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                    return len(content) > 0  # 可选：也可以加图片头部判断
            except Exception as e:
                print(f"[跳过无效图像] {path} 无法读取，原因：{e}")
                return False

        save_folder_path = f"./output/duckduckgo_images/{dataset_type}"
        os.makedirs(save_folder_path, exist_ok=True)

        for idx, sample in enumerate(data, start=1):
            text_query = sample['text'].strip('\'"')
            evidence_list = []
            uid = str(idx)
            image_folder_path = os.path.join(save_folder_path, uid)
            
            if evidence_cache:
                try:
                    image_files = sorted([
                        f for f in os.listdir(image_folder_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png")) and is_valid_image_file(os.path.join(image_folder_path, f))
                    ])
                    for img_file in image_files[:top_k]:
                        local_image_path = os.path.join(image_folder_path, img_file)
                        evidence_list.append(f"[Image Evidence]: {local_image_path}")
                        sample['evidence6'] = "\n\n".join(evidence_list) if evidence_list else ""
                    print(f"使用缓存图像：{image_folder_path}")
                except Exception as e:
                    print(f"读取本地图片失败：{image_folder_path}，错误：{e}")
            else:
                try:
                    evidence = get_evidence_images(text_query)
                    if isinstance(evidence, str):
                        try:
                            evidence_dict = json.loads(evidence)
                            if 'Information related to the text' in evidence_dict:
                                for entry in evidence_dict['Information related to the text']:
                                    if "Image:" in entry:
                                        image_part = entry.split("Image:")[1].strip()
                                        evidence_list.append(f"[Image Evidence]: {image_part}")
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON evidence: {e}")

                    if evidence_list:
                        image_folder_path = os.path.join(save_folder_path, str(idx))
                        os.makedirs(image_folder_path, exist_ok=True)

                        local_image_paths = [] 
                        for img_idx, image_url in enumerate(evidence_list[:top_k]):
                            local_image_path = download_image(image_url, image_folder_path, f"img{img_idx}")
                            if local_image_path:
                                local_image_paths.append(f"[Image Evidence]: {local_image_path}")

                        sample['evidence6'] = "\n\n".join(local_image_paths) if local_image_paths else ""
                    else:
                        sample['evidence6'] = "" 

                except Exception as e:
                    print(f"Error processing text query {text_query}: {e}")
                    sample['evidence6'] = ""


    if 7 in include_evidences:
        if not evidence_cache:
            print("Running question generation and text retrieval...")
            run_generation_pipeline(data, split=dataset_type) 
        text_queries = []
        for i in range(len(data)):
            file_path = f"./output/generated_questions/{dataset_type}/{i}/question_gen/annotation.json"
            if evidence_cache:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        gen_result = json.load(f)
                    text_query = gen_result.get("text_evidence_query", "")
                    uncertainty = gen_result.get("uncertainty", "")

                    data[i]['text_evidence_query'] = text_query
                    data[i]['uncertainty'] = uncertainty
                    text_queries.append(text_query if text_query else "")
                    print(f"使用生成缓存：{file_path}")
                except Exception as e:
                    print(f"读取生成缓存失败 index {i}: {e}")
                    data[i]['text_evidence_query'] = ""
                    data[i]['uncertainty'] = ""
                    text_queries.append("")
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        gen_result = json.load(f)

                        text_query = gen_result.get("text_evidence_query", "")
                        uncertainty = gen_result.get("uncertainty", "")

                        data[i]['text_evidence_query'] = text_query
                        data[i]['uncertainty'] = uncertainty
                        text_queries.append(text_query if text_query else "")

                except Exception as e:
                    print(f"Error loading annotation for index {i}: {e}")
                    data[i]['text_evidence_query'] = ""
                    data[i]['uncertainty'] = ""
                    text_queries.append("")

            save_folder_path = f"./output/caption2text/gen_question/{dataset_type}"
            os.makedirs(save_folder_path, exist_ok=True)

            api_key = os.getenv("GOOGLE_VISION_API_KEY")  
            cse_id = os.getenv("cse_id")
            text_search_query_num = 2
            do_domain_filter = False

            for i, text_query in enumerate(text_queries):
                if not text_query.strip():
                    data[i]['evidence7'] = []
                    continue

                uid = str(i)
                text_query = text_query.strip('\'"')
                annotation_file = os.path.join(save_folder_path, uid, "annotations.json")
                if evidence_cache:
                    try:
                        with open(annotation_file, "r", encoding="utf-8") as f:
                            annotations = json.load(f)
                        annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                        for ann in annotations:
                            ann['query'] = data[i].get('text_evidence_query', "")
                            ann['question'] = data[i].get('uncertainty', "")
                        data[i]['evidence7'] = annotations[:top_k] if annotations else []
                        print(f"使用检索缓存：{annotation_file}")
                    except Exception as e:
                        print(f"读取 evidence7 缓存失败 index {i}: {e}")
                        data[i]['evidence7'] = []
                else:
                    try:
                        annotations = get_text_using_caption(
                            text_query=text_query,
                            save_folder_path=save_folder_path,
                            api_key=api_key,
                            cse_id=cse_id,
                            text_search_query_num=text_search_query_num,
                            uid=uid,
                            do_domain_filter=do_domain_filter
                        )
                        if annotations:
                            annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                        for ann in annotations:
                            ann['query'] = data[i].get('text_evidence_query', "")
                            ann['question'] = data[i].get('uncertainty', "")
                        data[i]['evidence7'] = annotations[:top_k] if annotations else []

                        annotation_dir = os.path.join(save_folder_path, uid)
                        os.makedirs(annotation_dir, exist_ok=True)
                        with open(os.path.join(annotation_dir, "annotations.json"), "w", encoding="utf-8") as f:
                            json.dump(annotations, f, indent=2, ensure_ascii=False)

                    except Exception as e:
                        print(f"Error retrieving text for index {i}: {e}")
                        data[i]['evidence7'] = []
    

    if 8 in include_evidences:
        if not evidence_cache:
            print("Running question generation and image retrieval...")
            run_generation_pipeline(data, split=dataset_type)  

        caption = []
        for i in range(len(data)):
            file_path = f"./output/generated_questions/{dataset_type}/{i}/question_gen/annotation.json"
            if evidence_cache and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        gen_result = json.load(f)
                    question_gen = gen_result.get("image_evidence_query", "")
                    data[i]['image_evidence_query'] = question_gen
                    caption.append(question_gen if question_gen else "")
                    uncertainty = gen_result.get("uncertainty", "")
                    data[i]['uncertainty'] = uncertainty
                    print(f"使用生成缓存：{file_path}")
                except Exception as e:
                    print(f"读取生成缓存失败 index {i}: {e}")
                    caption.append("")
                    data[i]['image_evidence_query'] = ""
                    data[i]['uncertainty'] = ""
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        gen_result = json.load(f)
                    question_gen = gen_result.get("image_evidence_query", "")
                    data[i]['image_evidence_query'] = question_gen
                    caption.append(question_gen if question_gen else "")
                    uncertainty = gen_result.get("uncertainty", "")
                    data[i]['uncertainty'] = uncertainty
                except Exception as e:
                    print(f"Error loading annotation for index {i}: {e}")
                    caption.append("")
                    data[i]['image_evidence_query'] = ""
                    data[i]['uncertainty'] = ""

        save_folder_path = f"./output/gen_question/caption2image/{dataset_type}"
        os.makedirs(save_folder_path, exist_ok=True)

        api_key = os.getenv("GOOGLE_VISION_API_KEY")
        cse_id = os.getenv("cse_id")
        image_search_query_num = 1
        base64 = False

        for i, text_query in enumerate(caption):
            if not text_query.strip():
                data[i]['evidence8'] = []
                continue

            uid = str(i)
            text_query = text_query.strip('\'"')
            annotation_file = os.path.join(save_folder_path, uid, "annotations.json")

            if evidence_cache and os.path.isfile(annotation_file):
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        annotations = json.load(f)
                    annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                    for ann in annotations:
                        ann['query'] = data[i].get('image_evidence_query', "")
                        ann['question'] = data[i].get('uncertainty', "")
                    data[i]['evidence8'] = annotations[:top_k] if annotations else []
                    print(f"使用图像检索缓存：{annotation_file}")
                except Exception as e:
                    print(f"读取检索缓存失败 index {i}: {e}")
                    data[i]['evidence8'] = []
            else:
                try:
                    annotations = get_image_using_caption(
                        text_query=text_query,
                        save_folder_path=save_folder_path,
                        api_key=api_key,
                        cse_id=cse_id,
                        image_search_query_num=image_search_query_num,
                        uid=uid,
                        base64=base64
                    )
                    if annotations:
                        annotations = [ann for ann in annotations if not is_untrusted_evidence(ann, domain_blocklist)]
                    for ann in annotations:
                        ann['query'] = data[i].get('image_evidence_query', "")
                        ann['question'] = data[i].get('uncertainty', "")
                    data[i]['evidence8'] = annotations[:top_k] if annotations else []

                    annotation_dir = os.path.join(save_folder_path, uid)
                    os.makedirs(annotation_dir, exist_ok=True)
                    with open(os.path.join(annotation_dir, "annotations.json"), "w", encoding="utf-8") as f:
                        json.dump(annotations, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"Error retrieving image for index {i}: {e}")
                    data[i]['evidence8'] = []

    return data