#!/usr/bin/env python
# coding: utf-8

# In[3]:
import time 
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
import fasttext
from newspaper import Article
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.utils import encode_image


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def detect_web(path,how_many_queries,api_key):
    """Detects web annotations given an image."""
    client = vision.ImageAnnotatorClient(
        client_options={"api_key": api_key}
    )
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.web_detection(image=image, max_results=how_many_queries)
    annotations = response.web_detection
    return annotations

    
def get_inverse_search_annotation(web_annotations):
    file_save_counter = -1

    annotations = []
        
    for page in web_annotations.pages_with_matching_images: 
        file_save_counter = file_save_counter + 1
        new_entry = {}
        url = page.url
        domain = urlparse(url).netloc
        try:
            article = Article(url)
            article.download()
            article.parse()
            annotations.append({
                'title': str(page.page_title),
                'publish_date': str(article.publish_date),
                'text': str(article.text),
                'url': str(url),
                'domain': str(domain)
            })
        except Exception as e:
            print(f"Failed to parse article at {url}: {e}")
    return annotations 

def search_text_using_image(image_path, api_key, save_folder_path, text_search_query_num):
    all_results = []
    if isinstance(image_path, list):
        for path in image_path:
            result = detect_web(path,how_many_queries=text_search_query_num,api_key=api_key)
            inverse_search_results = get_inverse_search_annotation(result)
            all_results.extend(inverse_search_results)

    else:
        result = detect_web(image_path,how_many_queries=text_search_query_num,api_key=api_key)
        all_results = get_inverse_search_annotation(result)
    new_json_file_path = os.path.join(save_folder_path,'annotation.json')
    
    with io.open(new_json_file_path, 'w') as db_file:
        json.dump(all_results, db_file) 

    return all_results


def get_text_using_image(image_path, save_folder_path, api_key, text_search_query_num, uid, using_cached_evidence=False, mllm=None):
    image_text_dir = os.path.join(save_folder_path, str(uid), "image_text")
    image_text_json_file_path = os.path.join(image_text_dir, "annotation.json") 
    if using_cached_evidence and os.path.exists(image_text_dir) and os.path.exists(image_text_json_file_path):
        with open(image_text_json_file_path, 'r') as file:
            annotations = json.load(file)
    else:
        if os.path.exists(image_text_dir):
            shutil.rmtree(image_text_dir)
        os.makedirs(image_text_dir)
        annotations = search_text_using_image(image_path, api_key, image_text_dir, text_search_query_num)
    return annotations

