import re
import json
import time

from newspaper import Article
from ddgs import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from utils.predict_region import predict_region, pwarn


def text_search(query, query_type="title", top_k=5, retries=3):
    region = predict_region(query)
    max_results = 2 * top_k
    attempt = 0
    while attempt < retries:
        try:
            results = list(DDGS().text(query, region=region, safesearch='off', max_results=max_results))
            return results[:top_k] if results else []
        except DuckDuckGoSearchException as e:
            attempt += 1
            pwarn(f"Tool learning Warning: DuckDuckGo search failed on {query}. {e}. Retry {attempt}/{retries}.")
            time.sleep(5 * attempt)
    return []


def get_evidence_text(text, max_len=2000):
    top_k = 5
    results = text_search(text, query_type="text", top_k=top_k)
    retrieved_dict = {"Information related to the text": [text]}
    for evidence in results[:10]:
        info = f"Title: {evidence['title']}.\n Evidence: {evidence['body']}"
        retrieved_dict["Information related to the text"].append(info)
    return json.dumps(retrieved_dict)


def images_search(query, query_type="title", top_k=5, retries=3):
    region = predict_region(query)
    max_results = 2 * top_k
    attempt = 0
    while attempt < retries:
        try:
            results = list(DDGS().images(query, region=region, safesearch='off', max_results=max_results))
            return results[:top_k] if results else []
        except DuckDuckGoSearchException as e:
            attempt += 1
            pwarn(f"Tool learning Warning: DuckDuckGo search failed on {query}. {e}. Retry {attempt}/{retries}.")
            time.sleep(5 * attempt)
    return []


def get_evidence_images(text, max_len=2000):
    if not text.strip():
        return json.dumps({"error": "Query text is empty, cannot perform image search."})
    top_k = 5
    results = images_search(text, query_type="text", top_k=top_k)
    retrieved_dict = {"Information related to the text": [text]}
    if results:
        for evidence in results[:5]:
            info = f"Title: {evidence['title']}.\n Image: {evidence['image']}  \n  URL: {evidence['url']}"
            retrieved_dict["Information related to the text"].append(info)
    else:
        retrieved_dict["Information related to the text"].append("No search results found.")
    return json.dumps(retrieved_dict)


def news_search(query, query_type="title", top_k=5, retries=3):
    region = predict_region(query)
    max_results = 2 * top_k
    attempt = 0
    while attempt < retries:
        try:
            results = list(DDGS().news(query, region=region, safesearch='off', max_results=max_results))
            return results[:top_k] if results else []
        except DuckDuckGoSearchException as e:
            attempt += 1
            pwarn(f"Tool learning Warning: DuckDuckGo search failed on {query}. {e}. Retry {attempt}/{retries}.")
            time.sleep(5 * attempt)
    return []


def get_evidence_news(text, max_len=2000):
    if not text.strip():
        return json.dumps({"error": "Query text is empty, cannot perform news search."})
    top_k = 5
    results = news_search(text, query_type="text", top_k=top_k)
    retrieved_dict = {"Information related to the text": [text]}
    if results:
        for evidence in results[:5]:
            info = f"Body: {evidence['body']}"
            retrieved_dict["Information related to the text"].append(info)
    else:
        retrieved_dict["Information related to the text"].append("No search results found.")
    return json.dumps(retrieved_dict)
