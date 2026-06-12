# Chinese AF-EMR Named Entity Recognition

Code for named entity recognition (NER) on **Chinese atrial fibrillation (AF) electronic medical records (EMR)**, comparing a discriminative BERT-like model (**RoBERTa-BiLSTM-CRF**) with generative LLMs (**Qwen3-8B**, **ChatGLM3-6B**) adapted via LoRA / P-Tuning v2.

> ⚠️ **No raw clinical data is included.** All patient records, annotations, and intermediate datasets are withheld for privacy. Only code is published here. Each script expects you to supply your own data at the paths indicated at the top of the file.

## Entity categories

身体部位 (Body) · 症状体征 (Symptoms & Signs) · 检验检查 (Examinations & Tests) · 治疗方式 (Treatment) · 疾病和诊断 (Disease & Diagnosis)

## Models & key hyperparameters

The two generative models are fine-tuned with **different** parameter-efficient
methods (LoRA for Qwen3-8B, P-Tuning v2 for ChatGLM3-6B), so their learning
rates differ by design — each uses the rate recommended for its own method.

| Model | Method | Learning rate | Key settings |
|-------|--------|--------------:|--------------|
| RoBERTa-BiLSTM-CRF | full fine-tune | `3e-5` | Adam, seq_len 128, batch 16, 50 epochs, dropout 0.1 |
| Qwen3-8B | LoRA (8-bit) | `1.5e-4` | r=32, α=64, dropout 0.05, batch 8 × grad-accum 2 (eff. 16), seq_len 1024, 3 epochs |
| ChatGLM3-6B | P-Tuning v2 | `2e-2` | pre_seq_len 128, batch 1 × grad-accum 16 (eff. 16), max 3000 steps, 4-bit |

## Repository layout

```
training/
  roberta_bilstm_crf.py  RoBERTa-BiLSTM-CRF training + evaluation
  qwen_finetune.py       LoRA (8-bit) fine-tuning of Qwen3-8B for NER
  qwen_inference.py      Run the fine-tuned Qwen3-8B over a test set
  chatglm_ptuning.py     P-Tuning v2 fine-tuning of ChatGLM3-6B
  chatglm_inference.py   Run the P-Tuned ChatGLM3-6B over a test set

evaluation/
  compute_metrics.py     Per-category & micro P/R/F1 (strict span match)
  error_analysis.py      Micro-averaged metrics + per-category breakdown
  annotation_kappa.py    Inter-annotator agreement (Cohen's Kappa)

data_processing/
  ann_to_json.py         brat .ann  ->  JSON
  cut_json.py            Split JSON dataset
  merge_files.py         Merge annotation files
  auto_preannotate.py    Dictionary-based auto pre-annotation (human-corrected afterwards)
```

## Data format

Each sample is a chat-style record; the `assistant` content is the gold entity JSON:

```json
{
  "conversations": [
    {"role": "system",    "content": "<task instruction listing the 5 entity types>"},
    {"role": "user",      "content": "句子：<one EMR sentence>\n请按以下 JSON 模式输出：{...}"},
    {"role": "assistant", "content": "{\"身体部位\": [...], \"症状体征\": [...], ...}"}
  ]
}
```

See [`examples/sample.json`](examples/sample.json) for a complete worked example.
It is a **synthetic, illustrative** record (a constructed sentence) — **not real
patient data** — showing the system prompt, input, gold labels, and model output.

## Requirements

Python 3.10+. Main dependencies: `torch`, `transformers`, `peft`, `datasets`,
`modelscope`, `bitsandbytes`, `swanlab`, `pandas`, `pytorch-crf`.

```bash
pip install torch transformers peft datasets modelscope bitsandbytes swanlab pandas pytorch-crf
```

## Usage

1. Prepare your data in the JSON format above and set the path constants at the top of each script.
2. Fine-tune:  `python training/qwen_finetune.py`  (or `training/chatglm_ptuning.py`)
3. Infer:      `python training/qwen_inference.py`  (or `training/chatglm_inference.py`)
4. Evaluate:   `python evaluation/compute_metrics.py`

## Acknowledgements

The RoBERTa-BiLSTM-CRF implementation is adapted from
[jesska/Chinese-medical-entity-recognition(n)](https://github.com/jesska/Chinese-medical-entity-recognitionn).
We thank the authors for making their code available. Please also respect the
license of that repository when reusing the adapted code here.

## License

MIT

