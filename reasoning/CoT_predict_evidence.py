import dspy
from typing import List, Optional
import requests
import os
import ast
import re
from PIL import Image

def create_signature(selected_evidences):
    annotations = {
        "news_caption": str,
        "news_images": List[dspy.Image],
    }
    attrs = {
        "__doc__": "Verify the authenticity of news reports using multiple evidence sources.",
        "news_caption": dspy.InputField(desc="The news caption."),
        "news_images": dspy.InputField(desc="The news caption's accompanying images."),
    }
    if 1 in selected_evidences:
        annotations["Textual_Evidence_with_Image_search"] = List[str]
        attrs["Textual_Evidence_with_Image_search"] = dspy.InputField(
            desc="Retrieved Text Evidence from News Images: Text evidence retrieved by searching given news images. This text evidence is extracted from web pages exactly containing the query news images. If this text evidence aligns closely with the news caption, it demonstrates that the news image and the news caption are also highly consistent.",
            default=None
        )
    if 2 in selected_evidences:
        annotations["evidence2"] = List[dspy.Image]
        attrs["evidence2"] = dspy.InputField(
            desc="Retrieved Image Evidence from News Caption: Image evidence retrieved by searching the news caption. The search returns multiple web pages related to the caption, from which relevant images are extracted. If this image evidence aligns closely with the news images, it demonstrates that the news image and the news caption are moderately consistent.",
            default=None
        )
    if 3 in selected_evidences:
        annotations["Textual_Evidence_with_Caption_search"] = List[str]
        attrs["Textual_Evidence_with_Caption_search"] = dspy.InputField(
            desc="Retrieved Text Evidence from News Caption: Text evidence retrieved by searching the news caption. If this text evidence aligns closely with the news caption, it supports the authenticity of the caption.",
            default=None
        )
    if 4 in selected_evidences:
        annotations["evidence4"] = List[str]
        attrs["evidence4"] = dspy.InputField(
            desc="Retrieved Text Evidence from News Caption: Text evidence (from authoritative news websites) retrieved by searching the news caption. If this text evidence aligns closely with the news caption, it supports the authenticity of the caption.",
            default=None
        )
    if 5 in selected_evidences:
        annotations["evidence5"] = List[str]
        attrs["evidence5"] = dspy.InputField(
            desc="Retrieved Text Evidence from News Caption: Text evidence retrieved by searching the news caption. If this text evidence aligns closely with the news caption, it supports the authenticity of the caption.",
            default=None
        )
    if 6 in selected_evidences:
        annotations["evidence6"] = List[dspy.Image]
        attrs["evidence6"] = dspy.InputField(
            desc="Retrieved Image Evidence from News Caption: Image evidence retrieved by searching the news caption. The search returns multiple web pages related to the caption, from which relevant images are extracted. If this image evidence aligns closely with the news images, it demonstrates that the news image and the news caption are moderately consistent.",
            default=None
        )
    if 7 in selected_evidences:
        annotations["evidence7_question"] = str
        annotations["evidence7_query"] = str
        annotations["evidence7_evidence"] = str
        attrs["evidence7_question"] = dspy.InputField(
            desc="It could be a part or detail of the news story that you're not sure about, so you ask the following query to check",
             default=None
        )
        attrs["evidence7_query"] = dspy.InputField(
            desc="The query used to retrieve the evidence from search engine. This query is specially designed to search for evidence of possible misinformation in news posts.",
            default=None
        )
        attrs["evidence7_evidence"] = dspy.InputField(
            desc="Text Evidence retrieved by searching the specially designed query. This retrieved evidence gives contextual information to the potential misinformation targeted by the query.",
            default=None
        )
    
    if 8 in selected_evidences:
        annotations["evidence8_question"] = str
        annotations["evidence8_query"] = str
        annotations["evidence8_images"] = List[dspy.Image]
        attrs["evidence8_question"] = dspy.InputField(
            desc="It could be a part or detail of the news story that you're not sure about, so you ask the following query to check",
             default=None
        )
        attrs["evidence8_query"] = dspy.InputField(
            desc="The query used to retrieve the evidence from search engine. This query is specially designed to search for evidence of possible misinformation in news posts.",
            default=None
        )
        attrs["evidence8_images"] = dspy.InputField(
            desc="Image Evidence retrieved by searching the specially designed query. This retrieved evidence is the image related to the query.",
            default=None
        )
    annotations["label"] = bool
    annotations["confidence"] = int
    attrs["label"] = dspy.OutputField(
        desc="Whether the news report is authentic or not. If it is authentic, return True. Otherwise, return False."
    )
    attrs["confidence"] = dspy.OutputField(
        desc="The confidence score of your answer. It should be a integer between 0 and 100. The higher the score, the more confident your answer is."
    )
    
    attrs["__annotations__"] = annotations
    return type("Direct_Predict_Signature_Dynamic", (dspy.Signature,), attrs)


class CoT_Predict_Module_Evidence(dspy.Module):
    def __init__(self, include_evidences: list):
        Signature_Dynamic = create_signature(include_evidences)
        self.cot_predict_module = dspy.ChainOfThought(Signature_Dynamic)
        self.include_evidences = include_evidences
        self.sample_count = 0

    def forward(self, text, images, label=None, evidence1=None, evidence2=None, evidence3=None, 
                evidence4=None, evidence5=None, evidence6=None, evidence7=None,evidence8=None,evidence9=None):
        try:
            self.sample_count += 1 
            print("\n" + "="*50)
            print(f"Sample #{self.sample_count}")
            print("Input:")
            print(f"Text: {text}")
            print(f"Number of images: {len(images) if images else 0}")
            if label is not None:
                print(f"True Label: {label}")
            evidence_args = {}
            if 1 in self.include_evidences and evidence1 is not None:
                evidence_args['Textual_Evidence_with_Image_search'] = evidence1
            elif 1 in self.include_evidences:
                evidence_args['Textual_Evidence_with_Image_search'] = None

            if 2 in self.include_evidences and evidence2 is not None:
                evidence2_paths = []
                skipped_images = 0
                evidence2 = evidence2.split('\n\n')

                for item in evidence2:
                    match = re.search(r'\[local_image_path\]: (.+)', item)
                    if match:
                        local_image_path = match.group(1).strip()
                        if os.path.exists(local_image_path):
                            evidence2_paths.append(local_image_path)
                        else:
                            skipped_images += 1


                if skipped_images > 0:
                    print(f"Warning: Skipped {skipped_images} non-existent local image files")

                if evidence2_paths:
                    evidence_args['evidence2'] = evidence2_paths
                else:
                    evidence_args['evidence2'] = None

            elif 2 in self.include_evidences:
                evidence_args['evidence2'] = None
            
            if 3 in self.include_evidences and evidence3 is not None:
                evidence_args['Textual_Evidence_with_Caption_search'] = evidence3
            elif 3 in self.include_evidences:
                evidence_args['Textual_Evidence_with_Caption_search'] = None

            if 4 in self.include_evidences and evidence4 is not None:
                evidence_args['evidence4'] = evidence4
            elif 4 in self.include_evidences:
                evidence_args['evidence4'] = None

            if 5 in self.include_evidences and evidence5 is not None:
                evidence_args['evidence5'] = evidence5
            elif 5 in self.include_evidences:
                evidence_args['evidence5'] = None

            if 6 in self.include_evidences and evidence6 is not None:
                evidence6_paths = []
                skipped_images = 0
                evidence6 = evidence6.split('\n\n')

                for item in evidence6:
                    match = re.search(r'\[Image Evidence\]: (.+)', item)
                    if match:
                        local_image_path = match.group(1).strip()
                        if os.path.exists(local_image_path):
                            evidence6_paths.append(local_image_path)
                        else:
                            skipped_images += 1

                if skipped_images > 0:
                    print(f"Warning: Skipped {skipped_images} non-existent local image files")

                if evidence6_paths:
                    evidence_args['evidence6'] = evidence6_paths
                else:
                    evidence_args['evidence6'] = None

            elif 6 in self.include_evidences:
                evidence_args['evidence6'] = None

            if 7 in self.include_evidences and evidence7 is not None:
                evidence_args['evidence7_question'] = evidence7[0]['question']
                evidence_args['evidence7_query'] = evidence7[0]['query']
                evidence_args['evidence7_evidence'] = [ev['text'] for ev in evidence7]
            elif 7 in self.include_evidences:
                evidence_args['evidence7_question'] = None
                evidence_args['evidence7_query'] = None
                evidence_args['evidence7_evidence'] = None

            if evidence8 is not None and evidence8:
                evidence_args['evidence8_question'] = evidence8[0]['question']
                evidence_args['evidence8_query'] = evidence8[0]['query'] 
                evidence_args['evidence8_images'] = [ev['local_image_path'] for ev in evidence8] if evidence8 else None
            else:
                evidence_args['evidence8_question'] = None
                evidence_args['evidence8_query'] = None 
                evidence_args['evidence8_images'] = None

            if evidence9 is not None:
                evidence_args['evidence9'] = evidence9
                print(f"Evidence1 count: {len(evidence9)}")
            else:
                evidence_args['evidence9'] = None

            print(evidence_args)
            output = self.cot_predict_module(
                news_caption=text, 
                news_images=images,
                **evidence_args
            )
            
            if label is not None:
                is_correct = "✓ Correct" if output.label == label else "✗ Wrong"
                print("\nPrediction:")
                print(f"Reasoning:{output.reasoning}")
                print(f"Label: {output.label}")
                print(f"Confidence: {output.confidence}")
                print("\n")
                print(f"Result: {is_correct}")
            else:
                print("\nPrediction:")
                print(f"Reasoning:{output.reasoning}")
                print(f"Label: {output.label}")
                print(f"Confidence: {output.confidence}")
            
            print("="*50 + "\n")
            
            return output
        except Exception as e:
            print(f"Error: {e}")
            return None
