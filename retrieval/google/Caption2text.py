import argparse
import requests
import os
import PIL
import shutil
from PIL import Image
import imghdr
from bs4 import BeautifulSoup
import bs4
import time
from googleapiclient.discovery import build
from google.cloud import vision
import io
import os
from bs4 import NavigableString
import json
from utils.retrieval_utils import get_captions_from_page, save_html
from utils.utils import encode_image
from utils.retrieval_utils import compare_images
import time
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def google_search(search_term, api_key, cse_id, how_many_queries, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res_list = []
    for i in range(0, how_many_queries):
        start = i * 10 + 1
        res = service.cse().list(
            q=search_term,
            lr='lang_en',
            num=10,
            start=start,
            cx=cse_id,
            **kwargs
        ).execute()
        res_list.append(res)
    return res_list

"""
def parse_articles(search_results):
    articles = []
    for result in search_results:
        for item in result.get('items', []):
            url = item['link']
            try:
                article = Article(url)
                article.download()
                article.parse()
                articles.append({
                    'title': str(item['title']),
                    'publish_date': str(article.publish_date),
                    'text': str(article.text),
                    'url': str(url),
                    'domain': str(item['displayLink'])
                })
            except Exception as e:
                print(f"Failed to parse article at {url}: {e}")
    return articles
"""

def parse_articles(search_results):
    articles = []
    for result in search_results:
        for item in result.get('items', []):
            try:
                articles.append({
                    'title': str(item.get('title', '')),
                    'publish_date': '',  
                    'text': str(item.get('snippet', '')),
                    'url': str(item.get('link', '')),
                    'domain': str(item.get('displayLink', ''))
                })
            except Exception as e:
                print(f"Failed to process item {item}: {e}")
    return articles


def search_text_using_caption(text_query, save_folder_path, api_key, cse_id, text_search_query_num):
    result = google_search(text_query, api_key, cse_id, how_many_queries=text_search_query_num)
    search_results = parse_articles(result)
    new_json_file_path = os.path.join(save_folder_path,'annotation.json')
    with io.open(new_json_file_path, 'w') as db_file:
        json.dump(search_results, db_file) 
    return search_results


def get_text_using_caption(text_query, save_folder_path, api_key, cse_id, text_search_query_num, uid, using_cached_evidence=False, do_domain_filter=False, mllm=None, evidence_extraction=False):
    caption_text_dir = os.path.join(save_folder_path, str(uid), "caption_text")
    caption_text_json_file_path = os.path.join(caption_text_dir, "annotation.json") 
    
    if evidence_extraction:
        extracted_caption_text_json_file_path = os.path.join(caption_text_dir, "extracted_annotation.json") 

    if using_cached_evidence and os.path.exists(caption_text_dir) and os.path.exists(caption_text_json_file_path):
        with open(caption_text_json_file_path, 'r') as file:
            annotations = json.load(file)
    else:
        if os.path.exists(caption_text_dir):
            shutil.rmtree(caption_text_dir)
        os.makedirs(caption_text_dir)
        
        if text_query == "":
            annotations = []
        else:
            annotations = search_text_using_caption(text_query, caption_text_dir, api_key, cse_id, text_search_query_num)

    if not annotations or annotations == []:
        return []
    
    if evidence_extraction:
        if os.path.exists(extracted_caption_text_json_file_path):
            with open(extracted_caption_text_json_file_path, 'r') as file:
                annotations = json.load(file)
            annotations = domain_filter(annotations, do_domain_filter)
        else:
            annotations = run_evidence_extraction(mllm, caption=text_query, evidence=annotations)
            with io.open(extracted_caption_text_json_file_path, 'w') as db_file:
                json.dump(annotations, db_file) 

    return annotations