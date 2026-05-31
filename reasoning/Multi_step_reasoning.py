import dspy
from typing import List

class CheckCaptionAuthenticity(dspy.Signature):
    "Verify the authenticity of news caption."
    news_caption: str = dspy.InputField(desc="The news caption.")
    news_images: List[dspy.Image] = dspy.InputField(desc="The news caption's accompanying images.")
    caption_text_evidence: List[str] = dspy.InputField(desc="Text Evidence from News Caption: Text evidence retrieved by searching the news caption. If this text evidence aligns closely with the news caption, it supports the authenticity of the caption.")

    caption_label: bool = dspy.OutputField(desc="Whether the news caption is authentic or not. If it is authentic, return True. Otherwise, return False.")
    confidence: int = dspy.OutputField(desc="The confidence score of your answer. It should be a integer between 0 and 100. The higher the score, the more confident your answer is.")


class CaptionAuthModule(dspy.Module):
    def __init__(self):
        self.chain = dspy.ChainOfThought(CheckCaptionAuthenticity)

    def forward(self, caption, images, caption_text_evidence):
        return self.chain(
            news_caption=caption,
            news_images=images,
            caption_text_evidence=caption_text_evidence
        )


class CheckImageMisuse(dspy.Signature):
    "Verify the misuse of the image"
    news_caption: str = dspy.InputField(desc="The news caption.")
    news_images: List[dspy.Image] = dspy.InputField(desc="The news caption's accompanying images.")
    image_text_evidence: List[str] = dspy.InputField(desc="Text Evidence from News Images: Text evidence retrieved by searching given news images. This text evidence is extracted from web pages exactly containing the query news images. If this text evidence aligns closely with the news caption, it demonstrates that the news image and the news caption are also highly consistent.")

    image_misuse: bool = dspy.OutputField(desc="Whether the news image is misuse or not. If it is misuse, return True. Otherwise, return False.")
    confidence: int = dspy.OutputField(desc="The confidence score of your answer. It should be a integer between 0 and 100. The higher the score, the more confident your answer is.")


class ImageMisuseModule(dspy.Module):
    def __init__(self):
        self.chain = dspy.ChainOfThought(CheckImageMisuse)

    def forward(self, caption, images, image_text_evidence):
        return self.chain(
            news_caption=caption,
            news_images=images,
            image_text_evidence=image_text_evidence
        )


class FinalDecision(dspy.Signature):
    "Based on the judgment of the previous two steps, verify the authenticity of news report. Only label the overall news as False if the image misuse could lead readers to form a wrong impression about the news event. If the image is generic or only loosely relevant but does not cause misunderstanding, the news report may still be considered authentic."

    news_caption: str = dspy.InputField(desc="The news caption.")
    news_images: List[dspy.Image] = dspy.InputField(desc="The news caption's accompanying images.")
    caption_label: bool = dspy.InputField(desc="The predicted authenticity of the news caption from the previous step.")
    caption_confidence: int = dspy.InputField(desc="Confidence in the truthfulness of the news headline predicted in the previous step.")
    image_misuse: bool = dspy.InputField(desc="The predicted misuse status of the image from the previous step.")
    image_confidence: int = dspy.InputField(desc="Confidence in the image misuse state predicted in the previous step")
    reasoning_caption: str = dspy.InputField(desc="The explanation for the authenticity prediction of the caption.")
    reasoning_image: str = dspy.InputField(desc="The explanation for the misuse prediction of the image.")

    label: bool = dspy.OutputField(desc="Whether the news report is authentic or not. Return True only if the caption is authentic and the image is not misused. Otherwise, return False.")
    confidence: int = dspy.OutputField(desc="The confidence score of your answer. It should be a integer between 0 and 100. The higher the score, the more confident your answer is.")


class FinalDecisionModule(dspy.Module):
    def __init__(self):
        self.chain = dspy.ChainOfThought(FinalDecision)

    def forward(self, caption, images, cap_label, cap_conf, img_misuse, img_conf, cap_reason, img_reason):
        return self.chain(
            news_caption=caption,
            news_images=images,
            caption_label=cap_label,
            caption_confidence=cap_conf,
            image_misuse=img_misuse,
            image_confidence=img_conf,
            reasoning_caption=cap_reason,
            reasoning_image=img_reason
        )


class MultistepNewsChecker(dspy.Module):
    def __init__(self):
        self.caption_checker = CaptionAuthModule()
        self.image_checker = ImageMisuseModule()
        self.final_decider = FinalDecisionModule()
        self.sample_count = 0

    def forward(
        self,
        text: str,
        images: List[dspy.Image],
        evidence3: List[str],
        evidence1: List[str],
        label: bool
    ):
        try:
            self.sample_count += 1
            print(f"\n===== Sample #{self.sample_count} =====")
            print(f"Input Caption: {text}")
            print(f"Ture Label: {label}")

            cap_result = self.caption_checker.forward(
                caption=text,
                images=images,
                caption_text_evidence=evidence3
            )
            print("\n[Step 1: Caption Authenticity]")
            print(f"Predicted: {cap_result.caption_label}")
            print(f"Confidence: {cap_result.confidence}")
            print(f"Reasoning: {cap_result.reasoning}")

            img_result = self.image_checker.forward(
                caption=text,
                images=images,
                image_text_evidence=evidence1
            )
            print("\n[Step 2: Image Misuse Check]")
            print(f"Predicted misuse: {img_result.image_misuse}")
            print(f"Confidence: {img_result.confidence}")
            print(f"Reasoning: {img_result.reasoning}")

            final_result = self.final_decider.forward(
                caption=text,
                images=images,
                cap_label=cap_result.caption_label,
                cap_conf=cap_result.confidence,
                img_misuse=img_result.image_misuse,
                img_conf=img_result.confidence,
                cap_reason=cap_result.reasoning,
                img_reason=img_result.reasoning
            )
            print("\n[Final Decision]")
            print(f"Predicted Label: {final_result.label}")
            print(f"Confidence: {final_result.confidence}")
            print(f"Reasoning: {final_result.reasoning}")

            result_tag = "\u2713 Correct" if final_result.label == label else "\u2717 Wrong"
            print(f"\nFinal Result: {result_tag}")
            print("=" * 60)

            return final_result

        except Exception as e:
            print(f"Error: {e}")
            return None