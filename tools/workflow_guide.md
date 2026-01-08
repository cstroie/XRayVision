# MedGemma Pediatric Fine-Tuning - Complete Workflow

## Prerequisites

- Python 3.8+ installed locally
- Access to your radiology database
- Google account for Colab
- Google Drive with sufficient storage (plan for ~5-10GB depending on dataset size)

## Phase 1: Local Data Preparation (Your Computer)

### Step 1.1: Install Local Dependencies

```bash
pip install pillow tqdm
# Plus your database driver (psycopg2, mysql-connector-python, etc.)
```

### Step 1.2: Customize Export Script

1. Open `export_xray_data.py`
2. Update database connection settings (lines 35-42)
3. Modify the SQL query to match your database schema (lines 44-54)
4. Replace mock data with actual database call (lines 60-65)
5. Update the image copy logic (line 91)

### Step 1.3: Run Export

```bash
# Test with small sample first
python export_xray_data.py  # Edit script to uncomment limit=100

# Then full export
python export_xray_data.py
```

**Expected output:**
- `pediatric_xray_dataset/` folder
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- `images/` folder with PNG files
- `dataset_stats.json`

### Step 1.4: Preprocess Images

```bash
python preprocess_images.py
```

**Expected output:**
- `images_processed/` folder with resized images
- Updated JSONL files pointing to processed images

### Step 1.5: Verify Export

Check the dataset:
```bash
# Check file counts
ls pediatric_xray_dataset/images_processed/ | wc -l

# Check JSONL files
head -n 1 pediatric_xray_dataset/train.jsonl

# Review statistics
cat pediatric_xray_dataset/dataset_stats.json
```

## Phase 2: Upload to Google Drive

### Step 2.1: Upload Dataset

1. Open Google Drive in browser
2. Create folder: `My Drive/pediatric_xray_dataset`
3. Upload entire `pediatric_xray_dataset` folder (may take time)
4. Verify all files uploaded successfully

**Tip:** For large datasets, consider:
- Using Google Drive desktop app for faster upload
- Compressing as ZIP first, then unzipping in Colab
- Uploading in batches if connection is unstable

## Phase 3: Fine-Tuning in Google Colab

### Step 3.1: Create New Colab Notebook

1. Go to https://colab.research.google.com
2. File > New notebook
3. Copy code from `MedGemma_Pediatric_FineTune.ipynb`
4. Or upload the .ipynb file directly

### Step 3.2: Enable GPU

1. Runtime > Change runtime type
2. Hardware accelerator > T4 GPU
3. Save

### Step 3.3: Run Cells Sequentially

**Important:** Run cells one by one, especially first time:

1. **Cell 1:** Check GPU availability (must show GPU detected)
2. **Cell 2:** Install packages (~2-3 minutes)
3. **Cell 3:** Mount Google Drive (authorize access)
4. **Cell 4:** Import libraries
5. **Cell 5-6:** Load model (~5-10 minutes first time)
6. **Cell 7:** Configure LoRA
7. **Cell 8:** Load datasets (verify counts match your export)
8. **Cell 9:** Training config
9. **Cell 10:** **START TRAINING** (this will take longest)

### Step 3.4: Monitor Training

**Free Colab Limitations:**
- ~12 hours max session
- May disconnect randomly
- GPU not always available

**Training time estimates:**
- 100 samples: ~30 minutes
- 500 samples: ~2-3 hours
- 1000+ samples: ~4-8 hours (may need multiple sessions)

**If disconnected:**
- Training progress is lost
- Saved checkpoints remain in Drive
- Can resume from last checkpoint (modify code to load checkpoint)

### Step 3.5: Save Results

After training completes:
1. Cell 11 saves model to Drive automatically
2. Download the output folder to your local machine
3. Keep a backup!

## Phase 4: Testing and Evaluation

### Step 4.1: Test in Colab

Run Cell 12-13 to:
- Generate sample reports
- Compare with ground truth
- Calculate test set metrics

### Step 4.2: Local Inference (Optional)

To use the fine-tuned model locally:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained("google/medgemma-2b")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./medgemma_pediatric_finetuned")
processor = AutoProcessor.from_pretrained("./medgemma_pediatric_finetuned")

# Inference
image = Image.open("test_xray.png")
prompt = "Analyze this pediatric chest X-ray (age group: infant) and provide a detailed radiology report."
inputs = processor(text=prompt, images=image, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256)
report = processor.decode(outputs[0], skip_special_tokens=True)
print(report)
```

## Troubleshooting

### Common Issues

**1. "No GPU available" in Colab**
- Solution: Runtime > Change runtime type > Select GPU
- If GPU unavailable: Try later (free tier has limits)

**2. "Out of memory" error**
- Reduce batch size to 1
- Increase gradient_accumulation_steps
- Use smaller image size (256x256)

**3. Training too slow**
- Reduce training samples for testing
- Consider Colab Pro for better GPU
- Use smaller model variant if available

**4. Model outputs nonsense**
- Check if image paths are correct in JSONL
- Verify images load properly
- May need more training epochs
- Check learning rate (try 1e-4 or 5e-5)

**5. Colab disconnects during training**
- Keep browser tab active
- Use Colab Pro for longer sessions
- Save checkpoints frequently (reduce save_steps)

### Getting Help

If issues persist:
- Check Colab logs carefully
- Verify dataset format matches examples
- Start with tiny dataset (10-20 samples) to test pipeline
- Review HuggingFace PEFT documentation

## Next Steps

After successful fine-tuning:

1. **Evaluate thoroughly**: Test on diverse age groups
2. **Iterate**: Adjust hyperparameters if needed
3. **Expand**: Add more data or age groups
4. **Deploy**: Consider API deployment for production use
5. **Monitor**: Track model performance on real cases

## Tips for Success

- Start small: Test entire pipeline with 50-100 samples first
- Document everything: Note what works and what doesn't
- Version control: Keep track of different training runs
- Validate carefully: Have radiologists review generated reports
- Be patient: First run always has issues to debug

## Estimated Timeline

- Data export setup: 2-4 hours
- Data preprocessing: 1-2 hours
- First training run: 4-8 hours (mostly waiting)
- Debugging and iteration: Variable
- **Total**: Plan for 2-3 days for first successful training
