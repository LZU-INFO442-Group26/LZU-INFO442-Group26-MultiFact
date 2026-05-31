from .CoT_predict_evidence import CoT_Predict_Module_Evidence
from .Prompt_Ensembles_evidence import Ensemble_Predict_Module_Evidence
from .Multi_step_reasoning import MultistepNewsChecker
from .Self_Consistency import SelfConsistencyPredictor


def get_reasoning_model(approach,include_evidences):
    if approach == "cot_prompt_evidence":
        return CoT_Predict_Module_Evidence(include_evidences)
    elif approach == "prompt_ensembles_evidence":
        return Ensemble_Predict_Module_Evidence(include_evidences)
    elif approach == "multi_step":
        return MultistepNewsChecker()
    elif approach == "self_consistency":
        return SelfConsistencyPredictor(include_evidences)
    else:
        raise ValueError(f"Unsupported reasoning approach: {approach}")
