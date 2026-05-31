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


class Question_Generation_Signature(dspy.Signature):
    """
    Predict whether a news post (including some images and a caption) contains potential misinformation or includes details you are uncertain about.
    First, identify any parts or details in the news post that you are unsure of, would like to verify further, or suspect may be false or factually inaccurate, and additional evidence would help confirm the information.

    Then, list exactly two concise and focused search queries or phrases that you would use on a public search engine (e.g., Google) to gather evidence for the uncertain details.
    One query should be designed to retrieve textual evidence (for example, news reports, official statements, or documents) that can verify or refute the factual claims, and the other should be designed to retrieve images (recent or historical) that can visually confirm or challenge the details in the news post.

    Do not add any time references in your queries. Please generate English.
    """

    news_images: List[dspy.Image] = dspy.InputField(desc="The images of the news post.")
    news_caption: str = dspy.InputField(desc="The caption of the news post.")

    uncertainty: str = dspy.OutputField(desc="Any parts of the news post you are unsure of, would like to verify further, or suspect may be false or factually inaccurate.")
    text_evidence_query: str = dspy.OutputField(desc="The query designed to retrieve textual evidence (e.g., news articles, official statements) that can verify or refute the uncertain details.")
    image_evidence_query: str = dspy.OutputField(desc="The query designed to retrieve images that can visually confirm or challenge the uncertain details.")
    
    
class Question_Generation_Module(dspy.Module):
    def __init__(self):
        self.question_generation_module = dspy.Predict(Question_Generation_Signature)

    def forward(self, news_images, news_caption):
        try:
            output = self.question_generation_module(news_images=news_images, news_caption=news_caption)
            uncertainty = output.uncertainty
            text_evidence_query = output.text_evidence_query
            image_evidence_query = output.image_evidence_query
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            uncertainty = ""
            text_evidence_query = news_caption
            image_evidence_query = news_caption
        return uncertainty, text_evidence_query, image_evidence_query
    
    
def run_generation_pipeline(batch_data, split="dev"):
    model = Question_Generation_Module()

    for i, item in enumerate(tqdm.tqdm(batch_data)):
        uncertainty, text_evidence_query, image_evidence_query = model(item.get('images', []), item['text'])
        output_path = f"./output/generated_questions/{split}/{i}/question_gen/annotation.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump({
                "uncertainty": uncertainty,
                "text_evidence_query": text_evidence_query,
                "image_evidence_query": image_evidence_query
            }, file, ensure_ascii=False, indent=4)

        cost = sum([x['cost'] for x in base_llm.history if x['cost'] is not None])
        print(f"cost:{cost}")
