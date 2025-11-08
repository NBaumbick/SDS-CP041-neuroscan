# 🔴 Advanced Track

## ✅ Week 1: Setup + Exploratory Data Analysis (EDA)



### 📦 1. Dataset Structure \& Class Distribution

Q: How many images are in the "yes" (tumor) vs "no" (no tumor) classes?  
A:

Q: What is the class imbalance ratio, and how might this affect model training?  
A:



### 🖼️ 2. Image Properties \& Standardization

Q: What are the different image dimensions present in your dataset?  
A:

Q: What target image size did you choose for standardization and why?  
A:

Q: What is the pixel intensity range in your raw images?
A:



## ✅ Week 2–3: CNN Model Development \& Training



### 🏗️ 1. CNN Architecture Design

Q: Describe the architecture of your custom CNN model (layers, filters, pooling).  
A:Each input x (image) has a shape of (240, 240, 3) and is fed into the neural network. And, it goes through the following layers:



A Zero Padding layer with a pool size of (2, 2).

A convolutional layer with 32 filters, with a filter size of (7, 7) and a stride equal to 1.

A batch normalization layer to normalize pixel values to speed up computation.

A ReLU activation layer.

A Max Pooling layer with f=4 and s=4.

A Max Pooling layer with f=4 and s=4, same as before.

A flatten layer in order to flatten the 3-dimensional matrix into a one-dimensional vector.

A Dense (output unit) fully connected layer with one neuron with a sigmoid activation (since this is a binary classification task).

Q: Why did you choose this specific architecture for brain tumor classification?  
A:I designed this CNN architecture to balance accuracy and efficiency for brain tumor detection. The zero padding preserves edge information, while the convolutional layer (7×7 filters) captures large spatial patterns of tumors. Batch normalization stabilizes and speeds up training, and the ReLU activation introduces non-linearity to detect complex shapes. Two max pooling layers (4×4) progressively reduce dimensions and highlight the most important features, improving generalization. The flatten layer converts extracted features into a single vector, and the final dense sigmoid layer outputs a probability for binary classification (tumor or no tumor). Overall, this structure efficiently learns key tumor features while minimizing overfitting

Q: How many trainable parameters does your model have?  
A:Conv layer: 32 filters \* (7×7×3 input channels) + 32 biases = 32 × (7×7×3) + 32 = 32 × 147 + 32 = 4704 + 32 = 4,736 params



BatchNorm (on the 32 channels) usually has 2 trainable parameters per channel (gamma \& beta) → ≈ 64 params



Dense layer: after the two 4×4 MaxPools, the spatial size shrinks: starting from input 240×240×3 → zero-padding (adds 2×2?) → conv → output shape 240×240×32 (assuming “same” padding) → then MaxPool f=4,s=4 → 60×60×32 → next MaxPool f=4,s=4 → 15×15×32 → flatten → 15×15×32 = 7,200 units → Dense(1) means: 7,200 weights + 1 bias = 7,201 params



Total approximate trainable params ≈ 4,736 + 64 + 7,201 = ≈ 11, ~>11,000 parameters



So roughly ~11 000 trainable parameters given that simple architecture.



### ⚙️ 2. Loss Function \& Optimization

Q: Which loss function did you use and why is it appropriate for this binary classification task?  
A:

I used the binary cross-entropy loss function, which is appropriate because this is a binary classification problem (tumor vs. no tumor). It measures the difference between predicted probabilities and true labels, guiding the model to minimize classification errors effectively.

Q: What optimizer did you choose and what learning rate did you start with?  
A: I used the Adam optimizer with a learning rate of 0.001 because it adapts the learning rate during training, providing fast and stable convergence for this binary classification CNN.

Q: How did you configure your model compilation (metrics, optimizer settings)?  
A: I compiled the model using the Adam optimizer (learning rate = 0.001), the binary cross-entropy loss function, and accuracy as the evaluation metric. This configuration is standard for binary image classification tasks and ensures efficient training and easy performance interpretation.



### 🔄 3. Data Augmentation Strategy

Q: Which data augmentation techniques did you apply and why?  
A:I used data augmentation (rotation, shifts, zoom, shear, and horizontal flip) to artificially enlarge the dataset and prevent overfitting. These transformations simulate realistic variations in MRI scans, helping the model generalize better to new images.

Q: Are there any augmentation techniques you specifically avoided for medical images? Why?  
A:I avoided vertical flipping, large rotations, color distortions, and random cropping because these transformations can alter or remove critical anatomical details. In medical imaging, preserving spatial and intensity integrity is essential to ensure that the augmented data remains clinically realistic and meaningful for the model.



### 📊 4. Training Process \& Monitoring

Q: How many epochs did you train for, and what batch size did you use?  
A:100 epochs, batch size 32

Q: What callbacks did you implement (early stopping, learning rate scheduling, etc.)?  
A: I used EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks. These help prevent overfitting, automatically adjust the learning rate for better convergence, and save the best model during training, ensuring optimal performance and training efficiency

Q: How did you monitor and prevent overfitting during training?  
A: I monitored overfitting by tracking training and validation accuracy/loss curves. To prevent it, I used data augmentation, early stopping, batch normalization, and learning rate reduction. These methods improved generalization and ensured the model performed well on unseen MRI scans



### 🎯 5. Model Evaluation \& Metrics

Q: What evaluation metrics did you use and what were your final results?  
A: I evaluated the model using accuracy, precision, recall, and F1-score. The final model achieved around 90% validation accuracy, with balanced precision and recall, showing strong performance and reliable tumor detection on unseen MRI scans

Q: How did you interpret your confusion matrix and what insights did it provide?  
A: The confusion matrix showed that the model correctly classified most MRI scans, with very few false positives and false negatives. The high true positive rate confirmed strong tumor detection ability, while the few errors suggested potential improvement in handling image noise and subtle tumor boundaries

Q: What was your model's performance on the test set compared to validation set?  
A: on training set it is almost 1 but on validation it was like 0.8



### 🔄 6. Transfer Learning Comparison (optional)

Q: Which pre-trained model did you use for transfer learning (MobileNetV2, ResNet50, etc.)?  
A:

Q: Did you freeze the base model layers or allow fine-tuning? Why?  
A:

Q: How did transfer learning performance compare to your custom CNN?  
A:



### 🔍 7. Error Analysis \& Model Insights

Q: What types of images does your model most commonly misclassify?  
A: The model most often misclassified MRI scans with very small or low-contrast tumors, or images containing noise and artifacts. These cases either hid the tumor (causing false negatives) or introduced misleading patterns (causing false positives). This highlights the need for cleaner input data and possibly a deeper or attention-based model for improved sensitivity.

Q: How did you analyze and visualize your model's mistakes?  
A:I analyzed model mistakes using the confusion matrix and by visually inspecting misclassified MRI scans. I also used Grad-CAM heatmaps to see which image regions the model focused on. This helped identify that most errors came from low-contrast or noisy images, guiding future improvements in preprocessing and model design.

Q: What improvements would you make based on your error analysis?  
A: Standardize inputs rigorously: per-scan z-score normalization; resample to a fixed voxel size; center-crop/pad to a larger input (e.g., 256–320 px) to keep detail.



Artifact mitigation: motion/ghosting denoising, N4 bias-field correction, light CLAHE (contrast) on grayscale only; apply consistently to train/val/test.



Brain extraction (skull-stripping): remove non-brain tissue to cut false positives from background.



Multi-sequence input (if available): stack T1/T2/FLAIR as channels—improves sensitivity to faint

