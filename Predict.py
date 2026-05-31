# main.py
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

import dspy
from utils.read_data_dev_test import read_data
from utils.metrics import validate_answer
from retrieval.evidence_loader import load_evidence
from reasoning import get_reasoning_model

from retrieval.post_process.evidence_extraction_caption_text import Caption_Text_Evidence_Extraction_Module
from retrieval.post_process.evidence_extraction_image_text import Image_Text_Evidence_Extraction_Module

load_dotenv()

"""
Program Overview:
This program serves as the main entry point.

Usage:
1. Specify the program's runtime configuration via command-line arguments.
   - Parameter List:
     - `--llm_name`: Specify the name of the language model to use.
     - `--dataset_split`: Specify the dataset split.
     - `--include_evidences`: Specify the types of evidence to include.
     - `--reasoning_approach`: Specify the reasoning approach.
     - `--evidence_extraction_module`: Specify the evidence extraction module, supporting extraction for "caption2text" and "image2text" evidence.

Notes:
- Ensure that necessary API keys and environment variables are set before running the program.
- Log files will record detailed runtime information for debugging and result verification.
"""

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_llm(llm_name):
    from dspy import LM
    if llm_name.startswith("openai/"):
        api_key = os.getenv("OPENAI_API_KEY")
        return LM(llm_name, api_key=api_key)
    elif llm_name.startswith("gemini/"):
        api_key = os.getenv("GEMINI_API_KEY")
        return LM(llm_name, api_key=api_key)
    else:
        sglang_url = "http://127.0.0.1:7501/v1"
        return LM(f"openai{llm_name}", api_base=sglang_url, api_key="dummy")

def main(args):
    load_dotenv()

    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/{args.reasoning_approach}_{args.dataset_split}_{args.include_evidences}_{timestamp}.log'
    sys.stdout = Logger(log_file)

    llm = setup_llm(args.llm_name)
    dspy.configure(lm=llm)

    raw_data = read_data(args.data_path, split=args.dataset_split)
    raw_data=raw_data[0:6]
    raw_data = load_evidence(raw_data, include_evidences=args.include_evidences,filter_untrusted=args.filter_untrusted,top_k=args.top_k_evidence,dataset_type=args.dataset_split,evidence_cache=args.evidence_cache)

    if args.evidence_extraction == "caption_text":
        print("Calling caption text evidence extraction...")
        caption_text_extraction_module = Caption_Text_Evidence_Extraction_Module()

        for i, item in enumerate(raw_data):
            if 'evidence3' not in item or not isinstance(item['evidence3'], (str, list)):
                print(f"Skipping item {i} due to missing or malformed evidence3")
                continue
            if isinstance(item['evidence3'], str):
                documents = item['evidence3'].split('\n\n')
            else:
                documents = item['evidence3']
            try:
                extracted = caption_text_extraction_module.forward(
                    caption_query=item['text'],
                    image_query=item['images'],
                    documents=documents
                )
            except Exception as e:
                print(f"Error on item {i}: {e}")
                continue
            item['evidence3'] = "\n\n".join(extracted)

    elif args.evidence_extraction == "image_text":
        print("Calling image text evidence extraction...")
        image_text_extraction_module = Image_Text_Evidence_Extraction_Module()

        for i, item in enumerate(raw_data):
            if 'evidence1' not in item or not isinstance(item['evidence1'], str):
                print(f"Skipping item {i} due to missing or malformed evidence1")
                continue
            documents = [d.strip() for d in item['evidence1'].split('\n\n') if d.strip()]
            try:
                extracted = image_text_extraction_module.forward(
                    caption_query=item['text'],
                    image_query=item['images'],
                    documents=documents
                )
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
            item['evidence1'] = "\n\n".join(extracted)


    if args.reasoning_approach == "multi_step":
        datasets = [
            dspy.Example(**d).with_inputs('text', 'images', 'evidence1', 'evidence3', 'label') for d in raw_data
        ]
    else:
        datasets = [
            dspy.Example(**d).with_inputs('text', 'images', 'evidence1', 'evidence2', 'evidence3', 'evidence4', 'evidence5', 'evidence6', 'evidence7', 'evidence8', 'label') for d in raw_data
        ]

    model = get_reasoning_model(args.reasoning_approach, include_evidences=args.include_evidences)

    evaluator = dspy.Evaluate(
        devset=datasets,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=100,

        provide_traceback=True
    )
    evaluator(model, metric=validate_answer)

    total_cost = sum([x['cost'] for x in llm.history if x.get('cost') is not None])
    print(f"Total LLM cost: {total_cost}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_name", type=str, default="openai/gpt-4o"
    )
    parser.add_argument("--data_path", type=str, default="/projects/vig/hzy/XFacta")
    parser.add_argument(
        "--reasoning_approach",
        type=str,
        choices=["cot_prompt_evidence", "prompt_ensembles_evidence","self_consistency","multi_step"],
        default="cot_prompt_evidence",
    )
    parser.add_argument(
        "--dataset_split", type=str, choices=["dev", "test"], default="dev"
    )
    parser.add_argument("--include_evidences", type=int, nargs='+', default=[],
                       help="The types of evidence to be included correspond to six distinct categories, numbered 1 through 8, for example: --include_evidences 1 3 5.")
    parser.add_argument(
        "--evidence_extraction", type=str, choices=["caption_text", "image_text"], default=None, help="选择提取证据的类型：'caption_text' 或 'image_text'"
    )
    parser.add_argument("--filter_untrusted",action="store_true",
                       help="Whether to enable domain name filtering (default is not enabled). Adding this parameter indicates enabling the filtering."
    )
    parser.add_argument("--top_k_evidence",type=int,default=5,
                       help="The maximum number of items to retain for each type of evidence is 5, by default."
    )
    parser.add_argument('--evidence_cache', action="store_true",
                       help="Set to use cached evidence")
    args = parser.parse_args()

    main(args)
