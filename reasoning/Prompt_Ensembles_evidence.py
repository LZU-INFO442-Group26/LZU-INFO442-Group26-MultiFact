import dspy
from typing import List, Optional
import requests
import re
import json
import os

def create_variant_signature(doc_string, selected_evidences):
    """
    Create the signature with optional evidence fields if they are included.
    """
    annotations = {
        "news_caption": str,
        "news_images": List[dspy.Image],
    }
    attrs = {
        "__doc__": doc_string,
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
        attrs["evidence6_images"] = dspy.InputField(
            desc="Images from image search results (optional).",
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
    annotations["confidence"] = float
    attrs["label"] = dspy.OutputField(
        desc="Whether the news report is authentic or not. If it is authentic, return True. Otherwise, return False."
    )
    attrs["confidence"] = dspy.OutputField(
        desc="The confidence score of your answer. It should be a integer between 0 and 100. The higher the score, the more confident your answer is."
    )
    
    attrs["__annotations__"] = annotations
    return type("VariantSignature", (dspy.Signature,), attrs)


class VariantWithEvidence(dspy.Module):
    def __init__(self, doc_string, include_evidences: list):
        VariantSignature = create_variant_signature(doc_string, include_evidences)
        self.predictor = dspy.Predict(VariantSignature)
        self.include_evidences = include_evidences
        
    def forward(self, text, images, label=None, evidence1=None, evidence2=None, evidence3=None, 
                evidence4=None, evidence5=None, evidence6=None, evidence7=None,evidence8=None,evidence9=None):

        evidence_args = {}
       
        if 1 in self.include_evidences and evidence1 is not None:
            evidence_args['Textual_Evidence_with_Image_search'] = evidence1
        elif 1 in self.include_evidences:
            evidence_args['Textual_Evidence_with_Image_search'] = None

    
        if 2 in self.include_evidences and evidence2 is not None:
            evidence2_paths = []
            skipped_images = 0

            if isinstance(evidence2, list):
                evidence2 = "\n\n".join(evidence2)

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

            for item in evidence6:
                if 'local_image_path' in item:
                    image_path = item['local_image_path']
                    if os.path.exists(image_path):
                        try:
                            with Image.open(image_path) as img:
                                img.verify() 
                                evidence6_paths.append(image_path)
                        except (IOError, SyntaxError):
                            skipped_images += 1
                            print(f"Warning: Skipped invalid image {image_path}")
                    else:
                        skipped_images += 1
                        print(f"Warning: Skipped non-existent image {image_path}")

            if skipped_images > 0:
                print(f"Warning: Skipped {skipped_images} non-existent or invalid images")

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

        evidence_args = {
            k: v for k, v in evidence_args.items()
            if v not in [None, "", [], {}]
        }
        return self.predictor(
            news_caption=text,
            news_images=images,
            **evidence_args
        )


class Ensemble_Predict_Module_Evidence(dspy.Module):
    def __init__(self, include_evidences: list):
        self.include_evidences = include_evidences
        # Create multiple variants
        self.variants = [
            VariantWithEvidence("Verify the authenticity of news reports by analyzing the news caption and accompanying images.", include_evidences),
            VariantWithEvidence("Can the statement be confirmed as accurate? Yes or No?", include_evidences),
            VariantWithEvidence("Does evidence support the claim? Verify or Refute.", include_evidences),
            VariantWithEvidence("Is there validity to the assertion? Confirm or Deny.", include_evidences),
            VariantWithEvidence("Is this a factual statement? Affirmative or Negative?", include_evidences),
            VariantWithEvidence("Does this hold up under scrutiny? Valid or Invalid?", include_evidences)
        ]
        self.sample_count = 0
    
    def forward(self, text, images, label=None, evidence1=None, evidence2=None, evidence3=None, 
                evidence4=None, evidence5=None, evidence6=None, evidence7=None,evidence8=None,evidence9=None):

        try:
            self.sample_count += 1
            print("\n" + "="*60)
            print(f"Sample #{self.sample_count}")
            print("\nInput:")
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

                if isinstance(evidence2, list):
                    evidence2 = "\n\n".join(evidence2)

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

            print(evidence_args)
            print("-"*60)
            print("\nIndividual Predictions:")
            print(f"{'Variant':15} | {'Label':8} | {'Confidence':10}")
            print("-"*40)

            all_results = []

            for i, variant in enumerate(self.variants):
                variant_name = f"Variant{i}" if i > 0 else "DirectVariant"

                result = variant(
                    text=text,
                    images=images,
                    evidence1=evidence1,
                    evidence2=evidence2,
                    evidence3=evidence3,
                    evidence4=evidence4,
                    evidence5=evidence5,
                    evidence6=evidence6,
                    evidence7=evidence7,
                    evidence8=evidence8,
                    label=label
                )

                
                all_results.append({
                    "name": variant_name,
                    "label": result.label,
                    "confidence": result.confidence
                })
                print(f"{variant_name:15} | {str(result.label):8} | {result.confidence:.2f}")
            
            true_count = sum(1 for r in all_results if r["label"] is True)
            false_count = sum(1 for r in all_results if r["label"] is False)
            
            predictions = [1 if r["label"] else 0 for r in all_results]
            confidences = [r["confidence"] for r in all_results]
            if sum(confidences) > 0:
                weighted_sum = sum(pred * conf for pred, conf in zip(predictions, confidences))
                total_confidence = sum(confidences)
                final_label = weighted_sum / total_confidence > 0.5
                final_confidence = total_confidence / len(confidences)
            else:
                final_label = False
                final_confidence = 0.0
            
            print("\nEnsemble Result:")
            print(f"Label: {final_label}")
            print(f"Confidence: {final_confidence:.2f}")
            
            if label is not None:
                is_correct = "✓ Correct" if final_label == label else "✗ Wrong"
                print(f"Result: {is_correct}")
            
            print("="*60 + "\n")
            
            return dspy.Prediction(
                label=final_label,
                confidence=final_confidence
            )
            
        except Exception as e:
            print(f"Error in sample #{self.sample_count}: {e}")
            return dspy.Prediction(label=False, confidence=0.0)
