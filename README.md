# Deepfake Detection under Social Media Compression

**Raj Bhadani | Summer Internship 2026 | NIT Kurukshetra**

XceptionNet-based deepfake detector trained on FaceForensics++ with compression-aware augmentation.

## Results

| Metric | Value |
|--------|-------|
| Best Val AUC (c40) | 0.9957 |
| Val Accuracy | 95.00% |
| Target AUC | 0.85 |
| Dataset | FF++ Deepfakes, 200 videos, c40 compression |
| Model | XceptionNet (timm) |
| Epochs | 10 |

## Methodology

- **Dataset:** FaceForensics++ (FF++) — 200 fake (Deepfakes) + 200 real (YouTube), c40 compression
- **Augmentation:** JPEG compression (quality 40–90), Gaussian noise, horizontal flip — simulates WhatsApp/YouTube/Twitter pipelines
- **Model:** XceptionNet fine-tuned, 20.8M parameters, 2x Tesla T4 GPUs
- **Loss:** CrossEntropyLoss + Adam optimizer + CosineAnnealingLR

## Files

| File | Description |
|------|-------------|
| `deepfake-pipeline-nitk.ipynb` | Full training pipeline |
| `results.json` | Training history and final metrics |
| `training_curves.png` | Loss and AUC curves |
| `best_model.pth` | Trained weights (Google Drive — link below) |

## Model Weights

Too large for GitHub — download from Google Drive:  
`https://drive.google.com/drive/folders/1QCDtyWBZBiUCNe-iVCX3-R2zgTGwFrw0?usp=sharing`

## Next Steps

- [ ] Cross-dataset evaluation on Celeb-DF-v2 and DFDC-Preview
- [ ] Frequency-domain (DCT) feature branch
- [ ] Compression-robustness loss term
- [ ] Paper draft — IEEE CVPR Workshop / ICVGIP

## References

1. Rössler et al., "FaceForensics++", ICCV 2019
2. Zhang et al., "Face Forgery Detection Based on Fine-Grained Clues", IEEE TCSVT 2024
