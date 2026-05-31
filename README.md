# [XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs](https://arxiv.org/abs/2508.09999)

## 🧠 About
![Example image](./XFacta.png)
The rapid spread of multimodal misinformation on social media calls for more effective and robust detection methods. Recent advances using multimodal large language models (MLLMs) have shown potential in addressing this challenge. However, it remains unclear whether the bottleneck lies in evidence retrieval or in reasoning, which hinders further progress.
Existing benchmarks either contain outdated events—leading to evaluation bias due to discrepancies with current social media scenarios, since MLLMs may memorize those events—or are artificially synthetic, failing to reflect real-world misinformation patterns. Additionally, there is a lack of comprehensive analysis of MLLM-based model design strategies.
To address these issues, we introduce XFacta, a contemporary, real-world dataset designed for evaluating MLLM-based detectors. We propose a pipeline to automatically construct datasets based on current trending topics. We systematically evaluate various MLLM-based misinformation detection strategies, comparing models of different architectures and scales, and benchmarking them against existing methods. Our analysis provides useful insights for improving multimodal misinformation detection.


## 📂 Dataset

### Dataset download
Please check [this link](https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK?usp=drive_link) to download the dataset.

### Dataset structure

```
XFacta/
├── fake_sample/
│   ├── media/                     # Folder containing image batches
│   │   ├── batch1/                # Each contains 100 images
│   │   ├── ...
│   │   └── batch12/
│   ├── batch1.json                # Metadata for batch1
│   ├── ...
│   └── batch12.json
├── true_sample/
│   ├── media/
│   │   ├── batch1/
│   │   ├── ...
│   │   └── batch12/
│   ├── batch1.json
│   ├── ...
│   └── batch12.json
├── dev.json                       # Development split metadata
└── test.json                      # Test split metadata
```





## 🔧Preparation

### API configuration
If you need to use OpenAI, Gemini, Google Search, add the following to your '.env 'file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_VISION_API_KEY= your_google_vision_api_key_here
cse_id= your_custom_search_engine_id_here
```

### Installing dependencies
```bash
git clone 
conda create -n xfacta python=3.10
conda activate xfacta
cd Xfacta
pip install -r requirements.txt
```

### Installing dspy
To support multiple image inputs, our dspy library has been modified in the original version.
```bash
cd dspy
pip install .[dev]
```

### if using a local large model
```bash
###sglang
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.9.post2"

###qwen
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install qwen-vl-utils

git lfs install
###qwen-2b
git clone https://huggingface.co/Qwen/Qwen2-VL-2B
###qwen-7b
git clone https://huggingface.co/Qwen/Qwen2-VL-7B
###qwen-2b
#git clone https://huggingface.co/Qwen/Qwen2-VL-72B
```

## 🚀Getting Started
### Call
```bash
cd Xfacta
python Predict.py --llm_name xxx --data_path xxx --reasoning_approach xxx --dataset_split xxx --include_evidences xxx --evidence_extraction xxx --top_k_evidence xxx --filter_untrusted --evidence_cache
```

### Parameters
```
1. llm_name
    - openai/model name: If you use a model released by OpenAI, for example: openai/gpt-4o
    - gemini/model name: If you use a model released by Google, for example: gemini/flash-2.0
    - The parent directory path of the local model's weight file: if you deploy a local model

2. --data_path: The path where the dataset is located
    - "/projects/vig/hzy/XFacta": An example

2. --reasoning_approach: The choice of reasoning method determines how the model generates conclusions.
    - "cot_prompt_evidence"
    - "prompt_ensembles_evidence"
    - "self_consistency"
    - "multi_step"

3. --dataset_split: Choose the dataset split to run.
    - "dev"
    - "test"

4. --include_evidences: Specify the types of evidence to include.
    - 1: Extract text from images.
    - 2: Generate images based on captions.
    - 3: Generate text based on captions.
    - 4: Retrieve news from DuckDuckGo.
    - 5: Retrieve text from DuckDuckGo.
    - 6: Retrieve images from DuckDuckGo.
    - 7: Generate questions and search for text evidence based on the questions.
    - 8: Generate questions and search for image evidence based on the questions.

5. --evidence_extraction
    - image_text: Extract evidence type 1.
    - caption_text: Extract evidence type 3.
      (If you want to use --evidence_extraction, you must call the corresponding --include_evidences 1 or 3)

6. --top_k_evidence: The maximum number of entries to retain for each evidence type.
    - 5: Retain the top five entries for each evidence type.

7. --filter_untrusted: Whether to enable the untrusted source filtering mechanism (flag type).

8. --evidence_cache：Set to use cached evidence (flag type).
```



## 🏗️ Project structure
```
├── Readme.md                      # Help
├── logs                           # Log information
├── outputs                        # Evidence output file.
├── reasoning                      # Inference code.
│   ├── default.json
│   ├── __init__.py             
│   ├── CoT_predict_evidence.py        
│   ├── Multi_step_reasoning.py              
│   ├── Prompt_Ensembles_evidence.py              
│   └── Self_Consistency.py        
├── retrieval                      # Evidence retrieval module.
│   ├── duckduckgo                 # DuckDuckGo search engine module
│   │   └── evidence_search.py     
│   ├── google                     # Google search engine module
│   │   ├── Caption2image.py          
│   │   ├── Image2text.py
│   │   └── ...
│   └── evidence_loader.py         # Main function for evidence collection and loading
├── utils                          # Utility functions
│   ├── llm_info.py        
│   └── ...
├── .env                           # API configuration file
└── Predict.py                     # Main execution file
```

## 📖 Citation
```
@article{xiao2025xfacta,
  title={XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs},
  author={Xiao, Yuzhuo and Han, Zeyu and Wang, Yuhan and Jiang, Huaizu},
  journal={arXiv preprint arXiv:2508.09999},
  year={2025}
}
```


