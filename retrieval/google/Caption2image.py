import argparse
import requests
import os
import shutil
from PIL import Image
import imghdr
from googleapiclient.discovery import build
import io
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.utils import encode_image
from utils.retrieval_utils import compare_images
import numpy as np


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def google_search(search_term, api_key, cse_id, how_many_queries, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res_list = []
    for i in range(0, how_many_queries):
        start = i * 10 + 1
        res = service.cse().list(q=search_term, searchType='image', lr='lang_en', num=10, start=start, cx=cse_id, **kwargs).execute()
        res_list.append(res)
    return res_list


def download_and_save_image(image_url, save_folder_path, file_name):
    try:
        response = requests.get(image_url, stream=True, timeout=(60, 60))
        if response.status_code == 200:
            response.raw.decode_content = True
            image_path = os.path.join(save_folder_path, file_name + '.jpg')
            with open(image_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if imghdr.what(image_path).lower() == 'png':
                img_fix = Image.open(image_path)
                img_fix.convert('RGB').save(image_path)
            return 1
        else:
            return 0
    except:
        return 0


def get_direct_search_annotation(search_results_lists, save_folder_path):
    image_save_counter = 0
    images = []

    for one_result_list in search_results_lists:
        if 'items' in one_result_list.keys():
            for item in one_result_list['items']:
                image = {}
                if 'link' in item.keys():
                    image['img_link'] = item['link']
                if 'contextLink' in item['image'].keys():
                    image['page_link'] = item['image']['contextLink']
                if 'displayLink' in item.keys():
                    image['domain'] = item['displayLink']
                if 'snippet' in item.keys():
                    image['snippet'] = item['snippet']
                if 'entity' in item.keys():
                    image['entity'] = item['entity']

                # Download and save images
                download_status = download_and_save_image(item['link'], save_folder_path, str(image_save_counter))
                if download_status == 0:
                    continue

                image['image_path'] = os.path.join(save_folder_path, str(image_save_counter) + '.jpg')
                images.append(image)
                image_save_counter += 1

    return images


def search_image_using_caption(text_query, save_folder_path, api_key, cse_id, image_search_query_num):
    result = google_search(text_query, api_key, cse_id, how_many_queries=image_search_query_num)
    direct_search_results = get_direct_search_annotation(result, save_folder_path)
    new_json_file_path = os.path.join(save_folder_path, 'annotation.json')
    with io.open(new_json_file_path, 'w') as db_file:
        json.dump(direct_search_results, db_file)
    return direct_search_results


def get_image_using_caption(text_query, save_folder_path, api_key, cse_id, image_search_query_num, uid, using_cached_evidence=False, base64=True, do_domain_filter=False, mllm=None):
    caption_image_dir = os.path.join(save_folder_path, str(uid), "caption_image")
    caption_image_json_file_path = os.path.join(caption_image_dir, "annotation.json")

    if using_cached_evidence and os.path.exists(caption_image_dir) and os.path.exists(caption_image_json_file_path):
        with open(caption_image_json_file_path, 'r') as file:
            annotations = json.load(file)
    else:
        if os.path.exists(caption_image_dir):
            shutil.rmtree(caption_image_dir)
        os.makedirs(caption_image_dir)

        if text_query == "":
            annotations = []
        else:
            annotations = search_image_using_caption(text_query, caption_image_dir, api_key, cse_id, image_search_query_num)

    if not annotations:
        return []

    final_annotations = []
    for annotation in annotations:
        final_annotation = annotation.copy()

        # Ensure that the local image path is correctly added
        final_annotation['local_image_path'] = annotation['image_path']  # The path where the image was saved locally
        
        current_img_pil = Image.open(annotation['image_path'])
        if base64:
            encoded_image = encode_image(annotation['image_path'])
        else:
            encoded_image = annotation['image_path']

        final_annotation["image"] = encoded_image
        current_img = np.array(current_img_pil)

        for existing_img_info in final_annotations:
            existing_img_pil = Image.open(existing_img_info['image_path'])
            existing_img = np.array(existing_img_pil)
            if compare_images(existing_img, existing_img_pil, current_img, current_img_pil, cutoff=15):
                break
        else:
            final_annotations.append(final_annotation)

    return final_annotations
