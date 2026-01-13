---
base_model: google/medgemma-4b-it
library_name: transformers
model_name: medgemma-4b-it-ped
tags:
- generated_from_trainer
- sft
- trl
licence: apache-2.0
tags:
- generated_from_trainer
- sft
- trl
- medical
- radiology
- x-ray
datasets:
- costinstroie/xray-chest-ped-test
language:
- en
pipeline_tag: image-text-to-text
---

# Model Card for MedGemma-4B-IT-Ped

## Model Description

**MedGemma-4B-IT-Ped** is a fine-tuned version of [Google's MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it), specifically optimized for generating radiology reports for **pediatric chest X-rays**.

This model accepts an X-ray image and a text prompt, outputting a detailed radiological description. It was trained using the `costinstroie/xray-chest-ped-test` dataset.

## Intended Use & Limitations

**Intended Use:**
*   Assisting radiologists in generating preliminary reports for pediatric chest X-rays.
*   Educational purposes for medical students learning radiology terminology.

**Limitations:**
*   **Specialized Domain:** This model is trained exclusively on pediatric chest X-rays. Performance on adult chest X-rays or other body parts (e.g., extremities, head) is not guaranteed.
*   **Not a Diagnostic Tool:** This model is a research tool and should not be used as a sole diagnostic device. All outputs must be reviewed by a qualified medical professional.
*   **Dataset Bias:** The model reflects the characteristics and potential biases of the training dataset.

## Quick Start

To use this model, you need to load it with `AutoModelForImageTextToText` and use the associated processor. Unlike standard text models, this requires passing an image alongside the text prompt.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Load model and processor
model_id = "costinstroie/medgemma-4b-it-ped"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Prepare inputs (replace 'xray_image.jpg' with your image path)
image = Image.open("xray_image.jpg").convert("RGB")
prompt = "Generate a radiology report for this chest X-ray of a 5-year-old male."

# Format messages for the model
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
]

# Process inputs
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

# Generate report
outputs = model.generate(**inputs, max_new_tokens=500)
report = processor.decode(outputs[0], skip_special_tokens=True)

print(report)
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.26.2
- Transformers: 4.57.3
- Pytorch: 2.9.0+cu126
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```
