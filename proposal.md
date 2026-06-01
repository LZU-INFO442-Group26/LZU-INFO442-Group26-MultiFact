# INFO 442 — M1 Project Proposal

### Project Title

## MultiFact: Multimodal Misinformation Detection

---

## Team Members

| Name (English) | Name (Chinese) | Student ID | Role |
|---|---|---|---|
| Jianwen Han | 韩建汶 | 320230941911 | Team Lead & Exploratory Data Analysis |
| Chuhao Wang | 王楚皓 | 320230942401 | Data Acquisition & Data Preprocessing |
| Ke Meng | 孟可 | 320230942231 | Exploratory Data Analysis & NLP-based modelling |
| Zhirun Han | 韩知润 | 320230941921 | Visualization & Dashboard Deployment |

---

# 1. Domain and Motivation

Multimodal misinformation, which combines misleading text with manipulated or out-of-context images, has emerged as a pervasive and high-impact threat across social media platforms. It disseminates rapidly, distorts public opinion, erodes social trust, and poses significant risks to public health, political discourse, and global information integrity. Traditional detection approaches rely on unimodal analysis or handcrafted visual/textual features, which fail to capture the complex interplay between deceptive captions and misleading imagery.

While multimodal large language models (MLLMs) offer new potential for misinformation detection, current research suffers from critical limitations. Most existing datasets are outdated, synthetic, or not reflective of real-world patterns, enabling MLLMs to rely on memorization rather than evidence-based reasoning. Furthermore, there is a lack of systematic investigation into whether detection bottlenecks stem from evidence retrieval or reasoning, hindering the development of robust, generalizable detectors.

This project addresses these gaps by building a contemporary, real-world multimodal misinformation dataset and conducting controlled, systematic evaluations of MLLM-based pipelines. The work aims to identify optimal retrieval and reasoning strategies, uncover core bottlenecks, and deliver practical insights for future misinformation detection research.

---

# 2. Dataset Description

## Data Sources

This study utilizes a pre-compiled, real-world multimodal misinformation dataset. The dataset is curated from mainstream public social media platforms, where real news samples are collected from verified, authoritative media accounts, and misinformation samples are gathered from community-flagged content and professionally verified rumors. The collection period spans January 2024 to April 2025, capturing contemporary social media dynamics. The complete dataset is publicly accessible via Google Drive: https://drive.google.com/drive/folders/1SKH0k30ZFFYsvtMae3t5LfAZOWatL-mm.


As illustrated in the structure, the data is organized into separate directories for fake and real samples (fake_sample, real_sample), accompanied by standardized JSON files (dev.json, test.json) to facilitate immediate model evaluation.

## Data Processing Methods

The dataset is organized into a structured directory format to facilitate efficient access and processing. As illustrated in the file structure, the data is divided into fake_sample (misinformation) and true_sample (real news) directories. Within each category, multimodal content is stored in batched subdirectories (e.g., media/batch1/ containing image batches), with corresponding metadata provided in JSON files (e.g., batch1.json). Additionally, standard development and test splits are provided via dev.json and test.json respectively. 

```text
XFacta/
├── fake_sample/
│   ├── media/
│   │   ├── batch1/
│   │   ├── ...
│   │   └── batch12/
│   ├── batch1.json
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
├── dev.json
└── test.json
```
To convert this hierarchical directory structure into a standard CSV format, we are going to implement data preprocessing through the following four core steps:

🗺️ Step 1: Define Structure & Parsing Rules

First, we are going to map out the hierarchy of the raw data. We will identify the category folders under the root directory (such as fake_sample and true_sample) and analyze the internal schema of the batch metadata files (e.g., batch1.json). Our primary goal in this step is to establish a clear mapping logic between the image files and their corresponding JSON text records, ensuring that we can accurately extract the relevant content in the subsequent stages.

📂 Step 2: Directory Traversal & Data Extraction

Next, we are going to write a script to traverse the file system layer by layer. We will navigate from the root directory into each category folder and read the associated JSON metadata files. During this traversal, we are going to dynamically construct the complete relative or absolute path for each image. Simultaneously, we will extract the attributes linked to each image—such as the text content and source—directly from the JSON files.

🏷️ Step 3: Feature Construction & Label Assignment

This is the critical step where we flatten the data. We are going to combine the extracted image paths and text content into individual rows to form a unified record. At the same time, we will automatically assign binary classification labels based on the parent folder's name; for instance, samples located within fake_sample will be labeled as fake news, while those in true_sample will be labeled as real news. Additionally, we are going to retain the batch_id as a specific field to ensure future traceability.

💾 Step 4: Dataset Splitting & Formatted Export

Finally, to ensure valid model evaluation, we are going to cross-reference the generated data table with the original split files (dev.json and test.json). We will add a new column named split to precisely mark whether each row belongs to the training, validation, or testing set. Once this is done, we are going to perform deduplication and check for missing values, and then export the processed two-dimensional table uniformly as a CSV file, making it ready for direct use by downstream data loaders.


---

## Expected Dataset Sizes

The dataset contains a total of **2,400 balanced multimodal samples**, split evenly between authentic and misinformation content:
- 1,200 `real_sample` posts from verified authoritative media sources (BBC News, CNN, Fox News, etc.)
- 1,200 `fake_sample` posts from community-flagged and professionally verified rumors

For standardized model evaluation, the data is partitioned as follows:
- Training set: 2,160 samples
- Development set: 120 samples
- Test set: 120 samples

The data is organized into **12 batches** (batch1 to batch12), each containing 100 samples.

---



## Expected Features

Each sample in the dataset includes comprehensive, fine-grained metadata and annotations:

| Feature | Description |
|---|---|
| `sample_type` | `real_sample` or `fake_sample`, indicating the authenticity of the post |
| `batch_file` | The batch name (e.g., `batch1`, `batch9`) for organizational purposes |
| `tweet_id` | Unique identifier for each social media post |
| `text` | The full text caption accompanying the image(s) |
| `images` | File path(s) to the associated visual content (e.g., `./media/batchX/...`) |
| `label` | Ground truth label: `TRUE` (authentic) or `FALSE` (misinformation) |
| `author` | The username of the original post creator |
| `post_url` | The original URL of the social media post |
| `date_posted` | Timestamp of when the post was published |
| `topic` | High-level category (e.g., `Society`, `Politics`, `Nature`) |
| `error_category` | For fake samples only: type of misinformation, e.g., `Deepfakes`, `De-contextualized` |
| `flagging_tweet` | Text from the fact-checking/flagging tweet that debunks the claim |
| `flagging_tweet_authors` | The author(s) of the fact-checking/flagging content |

To ensure rigorous research standards, strict leakage mitigation measures are implemented, including excluding post-training data, isolating data sources, and aligning temporal distributions between real and fake samples. Annotation quality is guaranteed through double labeling, cross-validation, and evidence traceability. The project adheres to ethical guidelines by avoiding the collection of personal data, preventing the amplification of harmful content, and ensuring compliance with data usage policies.

---

# 3. Scientific Question

The main scientific question of this project is:

> Can a contemporary, real-world multimodal misinformation dataset reveal the dominant bottleneck between evidence retrieval and reasoning, and identify strategies that generalize to unseen real-world misinformation?

The project specifically investigates:

1. Whether a 2024–2025 real-world dataset eliminates memorization bias and forces MLLMs to rely on evidence-based reasoning, rather than learned historical patterns.
2. How different evidence retrieval strategies (text-to-text, image-to-text, hybrid) affect performance across Deepfake, out-of-context, and text-misleading misinformation types.
3. Whether structured multi-step reasoning outperforms standard CoT in detecting image misuse and producing interpretable judgments.
4. Whether semi-automatic closed-loop updates preserve dataset freshness and improve generalization to emerging misinformation patterns.

We aim to produce measurable insights using:

- Standard classification metrics (accuracy, precision, recall, F1) disaggregated by misinformation type
- Controlled ablation analysis comparing retrieval and reasoning contributions
- Cross-architecture evaluation across closed-source and open-source MLLMs
- Out-of-distribution testing on unseen real-world misinformation
- Human evaluation of reasoning interpretability and factual alignment
---

# 4. Preliminary Hypothesis

Based on existing literature, preliminary analysis of social media misinformation patterns, and early observations of MLLM behavior, this project proposes the following testable hypotheses:

#### Contemporary Data Reduces Memorization Bias
- If misinformation data is collected from January 2024 onward, then memorization-driven performance gains in MLLMs will be significantly reduced.
- If a contemporary real-world dataset is used, then MLLMs will rely more on evidence-based reasoning than recalled knowledge.
- If real non-synthetic misinformation is tested, then hidden reasoning limitations of MLLMs will become observable.

#### Hybrid Retrieval Improves Detection Performance
- If text-to-text (T→Et) and image-to-text (I→Et) retrieval are combined, then overall detection accuracy will be higher than any single strategy.
- If verifying real factual claims, then T→Et will outperform other retrieval methods.
- If detecting out-of-context or manipulated media, then I→Et will be the most effective strategy.
- If using LLM-generated queries, then retrieval performance will be worse than direct caption/image search.

#### Multi-Step Reasoning Outperforms CoT
- If structured multi-step reasoning is applied, then detection accuracy will be higher than vanilla CoT.
- If detecting image misuse, then multi-step reasoning will significantly outperform CoT.
- If using multi-step reasoning, then reasoning paths will be more interpretable and traceable.

#### Large MLLMs Achieve Better Zero-Shot Performance
- If using large-scale MLLMs (e.g., GPT-4o), then zero-shot accuracy will exceed smaller or fine-tuned models.
- If using open-source large MLLMs, then performance will be competitive but lower than closed-source models.
- If testing on real-world data, then zero-shot inference will generalize better than fine-tuning.

#### Semi-Automatic Update Preserves Freshness
- If a detection-in-the-loop pipeline is used, then the dataset can be continuously updated with new misinformation.
- If MLLM-assisted filtering is applied, then annotation effort will decrease without losing quality.
- If the dataset is regularly updated, then generalization to future misinformation will remain strong.
---

# 5. Team Roles

| Team Member | Responsibilities |
|---|---|
| Jianwen Han | Overall project leadership, research planning, experimental design, milestone management, GitHub maintenance, and final integration |
| Chuhao Wang | Data collection from social media, dataset construction, sample balancing, annotation quality control, and evidence curation |
| Ke Meng | Evidence retrieval implementation, reasoning strategy testing, multimodal model evaluation, and quantitative analysis |
| Zhirun Han | Data statistics, visual analysis, result visualization, presentation preparation, and documentation |

All team members will collaboratively contribute to:
- research writing,
- experimental discussion,
- result interpretation,
- presentation preparation,
- and final project refinement.
---

# 6. Preliminary Technical Plan

This project follows a tightly coupled, end-to-end pipeline centered on **dataset construction → evidence retrieval → structured reasoning → evaluation & iteration**, with minimal external models and strong module interdependence. All components are planned and conceptual; only preliminary exploratory experiments have been completed.

#### Evidence Retrieval Module
We will independently design and implement eight retrieval strategies. **Google Search** serves as the primary engine, with **DuckDuckGo** as a fair comparative baseline:
1. Text-to-Text (T→Et): Retrieve textual evidence from captions.
2. Text-to-Image (T→Ei): Retrieve visual evidence from captions.
3. Image-to-Text (I→Et): Retrieve textual evidence from post images.
4. Query→Ei: Generate retrieval queries via GPT-4o for image search.
5. Query→Et: Generate retrieval queries via GPT-4o for text search.
6–8: DuckDuckGo variants of T→Et, T→Ei, and news-specific search.

We also implement **domain filtering** and **GPT-4o-assisted evidence extraction** to reduce noise and improve relevance.

#### Reasoning Strategy Design
We will design and compare four reasoning paradigms across **GPT-4o, Gemini-2.0-Flash, Qwen-VL-7B, Qwen-VL-72B**:
1. Chain-of-Thought (CoT)
2. Prompt Ensembles
3. Self-Consistency Voting
4. Structured Multi-Step Reasoning (caption check → image validation → final decision)

#### Ablation Studies (Core Experiments)
1. **Ablation 1**: No evidence vs. full evidence — quantify retrieval necessity.
2. **Ablation 2**: Single-strategy vs. hybrid retrieval — validate T→Et + I→Et superiority.
3. **Ablation 3**: CoT vs. multi-step reasoning — test structured reasoning gains.
4. **Ablation 4**: Closed-source vs. open-source MLLMs — assess model scale effects.

#### Evaluation Metrics
- Overall Accuracy
- Precision, Recall, F1-score (per real/fake class)
- Prediction confidence analysis
- Per-type performance: Deepfake, Out-of-Context, Text Misleading
- Human evaluation of reasoning interpretability

#### Baseline Comparison
We compare against both traditional and MLLM-based state-of-the-art methods:
- Traditional detectors: SENs, Mocheg, HAMMER
- MLLM-based methods: Sniffer, MMFakeBench, LEM

#### Error Analysis
We conduct fine-grained error analysis focused on:
- Misclassification patterns across Deepfake, OOC, and Text Misleading
- Retrieval failure cases
- Reasoning inconsistency in complex samples

#### Semi-Automatic Update Framework
We develop a **detection-in-the-loop pipeline**: the best-performing MLLM assists human reviewers to screen and annotate new posts, enabling continuous dataset expansion and long-term temporal relevance.

Note: All above modules are in the conceptual stage and will be implemented independently.





---

# 7. Expected Deliverables

The project plans to deliver:

- A novel, contemporary real-world multimodal misinformation benchmark (MultiFact), the first evidence-grounded dataset for 2024–2025 social media content
- A balanced multimodal misinformation dataset with verified annotations and fine-grained misinformation type labels
- Complete data curation and preprocessing pipelines, including topic and visual alignment scripts
- Systematic exploratory data analysis (EDA) with at least 8 visualizations of misinformation types, topics, and temporal distributions
- Eight evidence retrieval strategy implementations and four reasoning paradigm designs
- Quantitative evaluations across multiple MLLMs (GPT-4o, Gemini-2.0-Flash, Qwen-VL-7B/72B) and baseline comparisons
- A semi-automatic detection-in-the-loop framework for continuous dataset updating
- A stakeholder-friendly demo or interactive dashboard for result visualization
- A final written report and a concise presentation summarizing all findings
---