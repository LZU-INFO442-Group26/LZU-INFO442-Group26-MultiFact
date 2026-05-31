import dspy
from typing import List
import os
import json
from dotenv import load_dotenv
import tqdm

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_llm = dspy.LM("gpt-4o", api_key=api_key)
dspy.configure(lm=base_llm)


class Summarize_Document_Signature(dspy.Signature):
    document: str = dspy.InputField(desc="The document to be summarized.")
    summary: str = dspy.OutputField(desc="The summary of the document.")


class Caption_Text_Evidence_Extraction_Signature(dspy.Signature):
    image_query: List[dspy.Image] = dspy.InputField(desc="The Image Query for your reference.")
    caption_query: str = dspy.InputField(desc="The Query for your reference.")
    documents: List[str] = dspy.InputField(desc="The documents list to be extracted which are retrieved from the Internet by searching the Query.")
    extracted_documents: List[str] = dspy.OutputField(desc='The extracted documents from the given documents. Make sure the number of extracted documents is the same as the number of given documents.')


class Caption_Text_Evidence_Extraction_Module(dspy.Module):
    def __init__(self):
        self.caption_text_evidence_extraction_module = dspy.Predict(Caption_Text_Evidence_Extraction_Signature)
        self.summarize_document_module = dspy.Predict(Summarize_Document_Signature)

    def forward(self, caption_query, image_query, documents):
        output_evidence = []
        batch_size = 3
        for j in range(0, len(documents), batch_size):
            batch_documents = documents[j:j + batch_size]
            batch_results = []
            try:
                batch_results = self.caption_text_evidence_extraction_module.forward(
                    caption_query=caption_query,
                    image_query=image_query,
                    documents=batch_documents
                ).extracted_documents
                assert len(batch_results) == len(batch_documents)
            except Exception as e:
                print(f"Error extracting evidence!")
                print(f"len(batch_documents): {len(batch_documents)}")
                print(f"len(batch_results): {len(batch_results)}")
                batch_results = batch_documents
            output_evidence.extend(batch_results)

        for i, evidence in enumerate(output_evidence):
            if evidence and len(evidence) > 600:
                try:
                    summary = self.summarize_document_module.forward(document=evidence).summary
                    output_evidence[i] = summary
                except Exception as e:
                    print(f"Error summarizing document: {str(e)}")
        print(type(output_evidence))
        return output_evidence


if __name__ == "__main__":

    data_path = "/projects/vig/hzy/XFacta"
    split = "dev"
    model = Caption_Text_Evidence_Extraction_Module()
    filename = f'{split}.json'
    full_path = os.path.join(data_path, filename)

    with open(full_path, 'r', encoding='utf-8') as file:
        batch_data = json.load(file)

    for idx in tqdm.tqdm(range(len(batch_data))):
        item = batch_data[idx]
        result_item = {
            'text': item['text'],
            'label': bool(item['label']),
            'images': item.get('images', []),
        }

        evidence_path = f"/home/han.zeyu/XFacta/retrieval/evidence/Vanilla/use_text_search_text/{split}/{idx + 1}/caption_text/annotation.json"
        output_path = f"/home/han.zeyu/XFacta/retrieval/evidence/Vanilla/captiontotext_extraction/{split}/{idx + 1}/caption_text/annotation.json"

        if not os.path.exists(evidence_path):
            print(f"Evidence path {evidence_path} does not exist")
            continue

        try:
            with open(evidence_path, 'r', encoding='utf-8') as ev_file:
                ev_data = json.load(ev_file)
        except json.JSONDecodeError:
            print(f"Error loading evidence file {evidence_path}")
            continue

        output_data = []
        if len(ev_data) == 0:
            print(f"No evidence found for item {idx}")
            continue

        documents_list = []
        for ev in ev_data:
            if isinstance(ev, dict):
                title_part = ev.get('title', '').strip()
                text_part = ev.get('text', '').strip()
                combined = (title_part + " " + text_part).strip()
                documents_list.append(combined)

        extracted_evidence = model.forward(result_item['text'], result_item['images'], documents_list)

        for original_ev, extracted_ev in zip(ev_data, extracted_evidence):
            if isinstance(original_ev, dict):
                original_ev['title'] = ""
                original_ev['text'] = extracted_ev
                output_data.append(original_ev)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, ensure_ascii=False, indent=4)

        cost = sum([x['cost'] for x in base_llm.history if x['cost'] is not None])
        print(f"Index {idx} processed, cost: {cost}")
