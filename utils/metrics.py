def validate_answer(example, pred, trace=None):
    example_label = str(example.get('label') if isinstance(example, dict) else example.label)
    pred_label = str(pred.get('label') if isinstance(pred, dict) else pred.label)
    
    example_label_bool = example_label.lower() == "true"
    pred_label_bool = pred_label.lower() == "true"
    
    return example_label_bool == pred_label_bool
