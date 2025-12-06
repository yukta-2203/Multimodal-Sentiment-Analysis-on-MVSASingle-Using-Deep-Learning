# Multimodal Sentiment Analysis on MVSA-Single (Image + Text Fusion)
This project implements a multimodal deep learning model combining **text features (BERT)** and **image features (ResNet-18)** to classify sentiment in the **MVSA-Single** dataset into three categories: **0 = Negative, 1 = Neutral, 2 = Positive**. The model uses the **text sentiment label** as the ground truth.

## Objective
Social media posts convey sentiment through both captions and images. This project investigates how combining **visual + textual** signals improves sentiment classification. The multimodal model achieves a **validation accuracy of 0.70**, outperforming text-only and image-only baselines.

## Model Architecture
**Text Encoder (BERT)**
- Model: `bert-base-uncased`
- Output: 768-dim CLS embedding

**Image Encoder (ResNet-18)**
- Pretrained on ImageNet
- Final FC layer replaced with `nn.Identity()`
- Output: 512-dim embedding

**Fusion Mechanism**
- Early fusion (concatenation)
- Combined vector size: **768 + 512 = 1280**

**Classifier**
```
Linear → ReLU → Dropout → Linear → Softmax(3 classes)
```

## Training Configuration
- Optimizer: AdamW  
- Learning Rate: 1e-4  
- Batch Size: 16  
- Loss: Weighted CrossEntropyLoss  
- Max Epochs: 15 (Early Stopping Patience = 3)  
- Tokenizer: BERT tokenizer (max_length = 64)  
- Image Augmentations: Resize, Crop, Horizontal Flip, ColorJitter, Normalize  
- Train/Validation Split: 80% / 20%  
- Dataset Used: ~4000 samples  

## Dataset Structure
```
mvsa_raw/MVSA_Single/
├── labelResultAll.txt
└── data/
    ├── <ID>.jpg
    └── <ID>.txt
```

## Results
- **Multimodal Validation Accuracy: 0.70**
- Fusion improves performance in ambiguous or sarcastic cases
- Text-only and image-only perform worse

## How to Run
1. Upload `mvsa.zip` to:
```
/content/mvsa.zip
```
2. Run the notebook:
```
multimodal_sentiment_analysis_project.ipynb
```
3. Training begins automatically, and the best model checkpoint is saved.

## Future Work
- Cross-modal attention fusion  
- Use CLIP for aligned visual–text embeddings  
- Ablation studies  
- Extend to MVSA-Multiple  
- Transformer-based multimodal fusion  

## Citation
```bibtex
@software{yukta2025multimodal,
  author  = {Yukta},
  title   = {Multimodal Sentiment Analysis on MVSA-Single Using Image--Text Fusion},
  year    = {2025},
  url     = {https://github.com/yukta-2203}
}
```
