import dspy
from typing import List
import os
import json
import traceback
from dotenv import load_dotenv
import tqdm

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_llm = dspy.LM("gpt-4o", api_key=api_key)
dspy.configure(lm=base_llm)


class Summarize_Document_Signature(dspy.Signature):
    """
    Summarize the given document comprehensively and concisely. Make sure all important entities, events, and other key information are included.
    """
    document: str = dspy.InputField(desc="The document to be summarized.")
    summary: str = dspy.OutputField(desc="The summary of the document.")


class Image_Text_Evidence_Extraction_Signature(dspy.Signature):
    """
    Extract fragments from documents that are relevant to the image query and caption query.
    For each document, you have the following rules:
    - only extract from the given documents and do not fabricate any non-existent content. 
    - if some document segments are relevant to any key information in the Image Query, directly quote these segments from the documents.
    - if the whole document is relevant to the Image Query, retain the whole document.
    - if it is irrelevant to Image Query, return an empty string 
    - Try to only include the relevant document part instead of returning the whole thing back. But do not be too strict and find a good balance.
    """

    image_query: List[dspy.Image] = dspy.InputField(desc="The Image Query for your reference.")
    caption_query:str=dspy.InputField(desc="The Caption Query for your reference.")
    documents: List[str] = dspy.InputField(desc="The documents list to be extracted which are retrieved from the Internet by searching the Image Query.")

    extracted_documents: List[str] = dspy.OutputField(desc='The extracted documents from the given documents. Make sure the number of extracted documents is the same as the number of given documents. Make sure all letters in the extracted documents comes from the given documents without any modification. Input example: ["a long document partially relevant to the image query", "an irrelevant document", "a long document highly relevant to the image query"] Output example: ["some quoted document segments", "", "the whole document"]')

class Image_Text_Evidence_Extraction_Module(dspy.Module):
    def __init__(self):
        self.image_text_evidence_extraction_module = dspy.Predict(Image_Text_Evidence_Extraction_Signature)
        self.summarize_document_module = dspy.Predict(Summarize_Document_Signature)

    def forward(self, image_query, caption_query,documents):
        output_evidence = []
        batch_size = 3

        for j in range(0, len(documents), batch_size):
            batch_documents = documents[j:j+batch_size]
            batch_results = []  # 先定义，确保总是有值
            try:
                result = self.image_text_evidence_extraction_module.forward(
                    image_query=image_query,
                    caption_query=caption_query,
                    documents=batch_documents
                )
                batch_results = result.extracted_documents

                if not isinstance(batch_results, list) or len(batch_results) != len(batch_documents):
                    raise ValueError("Invalid extracted_documents format or length mismatch.")

            except Exception as e:
                print(f"[ERROR] Evidence extraction failed at batch {j}~{j+batch_size}")
                print(f"Reason: {str(e)}")
                traceback.print_exc()
                batch_results = [""] * len(batch_documents)  # 保证长度一致，空字符串兜底

            output_evidence.extend(batch_results)

        # 进一步摘要处理长段内容
        for i, evidence in enumerate(output_evidence):
            if evidence and len(evidence) > 600:
                try:
                    summary = self.summarize_document_module.forward(document=evidence).summary
                    output_evidence[i] = summary
                except Exception as e:
                    print(f"[ERROR] Failed to summarize document {i}: {str(e)}")
                    traceback.print_exc()

        return output_evidence




if __name__ == "__main__":
    data_path = "/work/vig/hzy/XFacta"
    splits = ["test"]
    model = Image_Text_Evidence_Extraction_Module()
    for split in splits:
        filename = f'{split}.json'
        full_path = os.path.join(data_path, filename)
        with open(full_path, 'r', encoding='utf-8') as file:
            batch_data = json.load(file)

        # Build result data first
        result_data = []
        for i, item in enumerate(batch_data):
            result_data.append({
                'text': item['text'],
                'label': bool(item['label']),
                'images':  item.get('images', []),
            })
        for i, item in enumerate(tqdm.tqdm(result_data)):
            evidence_path = f"XFacta/retrieval/evidence/Vanilla/use_images_search_text/{split}/{i}/image_text/annotation.json"
            output_path = f"XFacta/retrieval/evidence/Vanilla/imagetotest_extraction/{split}/{i}/image_text/annotation.json"
            if not os.path.exists(evidence_path):
                print(f"Evidence path {evidence_path} does not exist")
                continue
            with open(evidence_path, 'r', encoding='utf-8') as ev_file:
                try:
                    ev_data = json.load(ev_file)
                except json.JSONDecodeError:
                    print(f"Error loading evidence file {evidence_path}")
                    continue

            output_data = []
            if len(ev_data) == 0:
                print(f"No evidence found for item {i}")
                output_data = []
            else:
                documents_list = []
                for ev in ev_data:
                    if isinstance(ev, dict):
                        title_part = ev.get('title', '').strip()
                        text_part = ev.get('text', '').strip()
                        combined = (title_part + " " + text_part).strip()
                        documents_list.append(combined)

                extracted_evidence = model.forward(item['images'], item['text'], documents_list)
                for original_ev, extracted_ev in zip(ev_data, extracted_evidence):
                    if isinstance(original_ev, dict):
                        original_ev['title'] = ""
                        original_ev['text'] = extracted_ev
                        output_data.append(original_ev)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(output_data, output_file, ensure_ascii=False, indent=4)
