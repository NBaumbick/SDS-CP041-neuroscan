# 🔴 Advanced Track

## ✅ Week 1: Setup + Exploratory Data Analysis (EDA)


### 📦 1. Dataset Structure & Class Distribution

**Q: How many images are in the "yes" (tumor) vs "no" (no tumor) classes?**  
**A:** The dataset contains **155 tumor (yes)** images and **98 non-tumor (no)** images, totaling **253 images**.

**Q: What is the class imbalance ratio, and how might this affect model training?**  
**A:** The ratio is approximately **1.6:1 (yes:no)**, showing a mild imbalance toward tumor images. This can be handled later with augmentation or class weighting.


### 🖼️ 2. Image Properties & Standardization

**Q: What are the different image dimensions present in your dataset?**  
**A:** **No class:** width 150 – 1920 px, height 168 – 1080 px  
      **Yes class:** width 178 – 1275 px, height 173 – 1427 px  
<br>Average image size across the dataset is roughly **350 × 380 pixels**, confirming variability in original image resolutions.


**Q: What target image size did you choose for standardization and why?**  
**A:** All images were resized to **(128 × 128)** pixels to balance detail retention and computational efficiency.

**Q: What is the pixel intensity range in your raw images?**  
**A:** Grayscale pixel values range from **0 to 255**, and were normalized to **[0, 1]** during preprocessing for stable model training.


## ✅ **Summary:**  
The dataset has **253 MRI scans** (155 with tumors, 98 without).  
Images were resized, normalized, and verified for RGB consistency — ready for CNN model development in Week 2.


## ✅ Week 2–3: CNN Model Development & Training


### 🏗️ 1. CNN Architecture Design

**Q: Describe the architecture of your custom CNN model (layers, filters, pooling).**  
**A:** I built a sequential CNN with **three convolutional blocks** (filters = 32, 64, 128), each followed by **ReLU activation**, **batch normalization**, and **MaxPooling2D**.  
The extracted features were flattened and passed through **Dense(128, ReLU)** and **Dropout(0.5)** layers before the **sigmoid output layer** for binary classification.  

**Q: Why did you choose this specific architecture for brain tumor classification?**  
**A:** This structure balances depth and regularization—deep enough to capture fine-grained texture patterns in MRI scans while remaining lightweight to prevent overfitting on a 253-image dataset.  

**Q: How many trainable parameters does your model have?**  
**A:** The custom CNN contains approximately **1.1 million trainable parameters**, providing sufficient capacity for effective feature extraction without excessive computation.  


### ⚙️ 2. Loss Function & Optimization

**Q: Which loss function did you use and why is it appropriate for this binary classification task?**  
**A:** I used **Binary Cross-Entropy**, ideal for two-class problems as it measures the divergence between true labels and predicted probabilities.  

**Q: What optimizer did you choose and what learning rate did you start with?**  
**A:** The model used the **Adam optimizer** with an initial learning rate of **1e-3** for adaptive gradient updates and fast early convergence.  

**Q: How did you configure your model compilation (metrics, optimizer settings)?**  
**A:** The model was compiled using:  
```python
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy', 'Precision', 'Recall']
```
This setup ensures accurate monitoring of both correctness and sensitivity during tumor detection.

### 🔄 3. Data Augmentation Strategy

**Q: Which data augmentation techniques did you apply and why?**  
**A:** I applied **rotation (±30°)**, **zoom (±20%)**, **horizontal and vertical flips**, **brightness adjustments [0.6–1.4]**, and **shear/shift transformations**.  
These augmentations synthetically increased dataset diversity, improved generalization, and helped the model handle real-world MRI variability without overfitting.  

**Q: Are there any augmentation techniques you specifically avoided for medical images? Why?**  
**A:** Yes, I avoided **excessive geometric warping**, **vertical inversion**, and **color jittering**, as these can distort anatomical structures in MRI images and mislead the model during training.  


### 📊 4. Training Process & Monitoring

**Q: How many epochs did you train for, and what batch size did you use?**  
**A:** Training ran for **20–25 epochs** with a **batch size of 16**, ensuring balanced convergence without overfitting.  

**Q: What callbacks did you implement (early stopping, learning rate scheduling, etc.)?**  
**A:** I used **EarlyStopping**, **ReduceLROnPlateau**, and **ModelCheckpoint** to stop training when validation loss plateaued and automatically save the best-performing weights.  

**Q: How did you monitor and prevent overfitting during training?**  
**A:** Overfitting was mitigated using **dropout layers**, **real-time validation monitoring**, and **data augmentation**. The best checkpoint was restored using early stopping for stable generalization.  


### 🎯 5. Model Evaluation & Metrics

**Q: What evaluation metrics did you use and what were your final results?**  
**A:** Evaluation was based on **Accuracy, Precision, Recall, and F1-score**.  
- **MobileNetV2 (v1)**: 80% accuracy, F1 ≈ 0.85  
- **EfficientNetB0 (fixed)**: 92% accuracy, F1 ≈ 0.94  

**Q: How did you interpret your confusion matrix and what insights did it provide?**  
**A:** The final model correctly classified **46 out of 50 validation images**, with only **1 false negative**, showing high recall and reliable tumor detection.  

**Q: What was your model's performance on the test set compared to validation set?**  
**A:** The model achieved **~90%+ consistency across test and validation sets**, confirming strong generalization and effective overfitting control.  


### 🔄 6. Transfer Learning Comparison (optional)

**Q: Which pre-trained model did you use for transfer learning (MobileNetV2, ResNet50, etc.)?**  
**A:** I used **MobileNetV2** and **EfficientNetB0** architectures initialized with **ImageNet weights**.  

**Q: Did you freeze the base model layers or allow fine-tuning? Why?**  
**A:** Initially froze all convolutional layers, then **unfroze the top 20%** for controlled fine-tuning with a **1e-5 learning rate**, enhancing adaptability without destabilizing pretrained features.  

**Q: How did transfer learning performance compare to your custom CNN?**  
**A:** Transfer learning achieved significant gains:  
- Custom CNN: ~65% accuracy  
- MobileNetV2: 80% accuracy  
- EfficientNetB0 (fixed): 92% accuracy  
This confirmed the value of pretrained visual features for limited medical datasets.  


### 🔍 7. Error Analysis & Model Insights

**Q: What types of images does your model most commonly misclassify?**  
**A:** Errors mainly occurred on **low-contrast or peripheral tumor scans**, where tumor boundaries blended into normal tissue.  

**Q: How did you analyze and visualize your model's mistakes?**  
**A:** I used **confusion matrices** and **misclassified sample visualizations** to identify patterns, helping refine augmentation and preprocessing strategies.  

**Q: What improvements would you make based on your error analysis?**  
**A:** Future improvements include **contrast enhancement**, **larger medical dataset fine-tuning**, and **ensembling MobileNetV2 with EfficientNetB0** for added robustness.  


## ✅ **Summary:**  
Accuracy improved from **80% (MobileNetV2)** to **92% (EfficientNetB0 corrected)** through better preprocessing, augmentation, and class balancing.  
The final EfficientNetB0 model demonstrated **high precision (0.91)** and **recall (0.97)** — achieving reliable tumor detection on limited MRI data.
