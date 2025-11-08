# 🔴 Advanced Track

## ✅ Week 1: Setup + Exploratory Data Analysis (EDA)


### 📦 1. Dataset Structure & Class Distribution

Q: How many images are in the "yes" (tumor) vs "no" (no tumor) classes?  
A: yes: 155, no: 98 -> After preprocessing and duplication removal we have yes: 129, no: 76

Q: What is the class imbalance ratio, and how might this affect model training?  
A: 98/155 ~> 0.63. This imbalance could lead to a "yes" bias. In other words, the model could become slightly better at identifying 
   images with tumors than it is at identifying images without tumors. After preprocessing we have 76/129 ~>  0.6


### 🖼️ 2. Image Properties & Standardization

Q: What are the different image dimensions present in your dataset?  
A: The no images width ranges from 150-1920 where the height ranges from 168-1080
   while the "yes" images range from 178-1275 width and 173-1275 height.

Q: What target image size did you choose for standardization and why?  
A: 224x224 to retain as much detail as possible since some images will be downsized by a factor of 5.
   I think this option makes sense as it could reduce noise in the majority while also making the dataset 
   images the same size as the ones used for ResNet. There is the issue of about 20% of the image width's 
   being below the 244 resolution mark which could introduce some noise in the upscale but the alternative
   of downscaling every image to 150x150 would lose a significant amount of data.

Q: What is the pixel intensity range in your raw images?
A: The pixel intensity range is from 0-255



## ✅ Week 2–3: CNN Model Development & Training


### 🏗️ 1. CNN Architecture Design

Q: Describe the architecture of your custom CNN model (layers, filters, pooling).  
A: The model has 3 convolutional -> pooling layers, each separated by a normalization layer. Following this 
   are a flattening layer, a dense layer, a dropout layer and finally another dense layer for the output.
   

Q: Why did you choose this specific architecture for brain tumor classification?  
A: TensorFlow was chosen over PyTorch simply because I've built models with PyTorch before but not 
   so much with TensorFlow so I wanted to switch things up. Seems to have a lot of functionality integrated
   already so that's nice. The architecture was chosen after several iterations of refinement on the model demonstrated by
   https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks

Q: How many trainable parameters does your model have?  
A:  The model has 38,815,749 total parameters with 12,938,433 trainable parameters.

### ⚙️ 2. Loss Function & Optimization

Q: Which loss function did you use and why is it appropriate for this binary classification task?  
A:  Binary cross-entropy with a sigmoid output as specified by the instructions.

Q: What optimizer did you choose and what learning rate did you start with?  
A:  Adam as specified by the instructions again with a starting learning rate of 1e-4 (chosen from standard
    starting LR for LoRa) Added ReduceLROnPlateau to try to help with EarlyStopping callback.

Q: How did you configure your model compilation (metrics, optimizer settings)?  
A:  Model configured with several metrics, including accuracy, precisions, recall, auc, True Positives, False Positives, True Negatives, and False Negatives.


### 🔄 3. Data Augmentation Strategy

Q: Which data augmentation techniques did you apply and why?  
A:  Random rotations, zoom, and a horizontal flip were added to create variance without breaking
    the anatomy.

Q: Are there any augmentation techniques you specifically avoided for medical images? Why?  
A: Vertical flipping and larger rotations as they might distort the intensity patterns.


### 📊 4. Training Process & Monitoring

Q: How many epochs did you train for, and what batch size did you use?  
A:  I set a batch size of 32 for 100 epochs, however the training never seemed to make it through the full
    set of epochs as EarlyStopping was implemented.

Q: What callbacks did you implement (early stopping, learning rate scheduling, etc.)?  
A:  I added early stopping, model checkpointing and reduce LR on plateau (as mentioned early to help 
    with the early stopping). All of these were set to monitor AUC after some testing as it is more stable 
	with only a few samples.

Q: How did you monitor and prevent overfitting during training?  
A: A lot of trial and error utilizing the graphs,
   Added in L2 regularization weight decays to prevent the model from relying on memorized weights,
   Dropout,
   Early stopping on the loss value so that the model doesn't continue to fit training noise after it
   has the "best checkpoint",
   Performed the same normalization on both testing and validation sets.
   Finally, I added shuffling to both the training and validation sets.


### 🎯 5. Model Evaluation & Metrics

Q: What evaluation metrics did you use and what were your final results?  
A: I tracked accuracy, precision, recall, and AUC across 100 training epochs. 
Final validation results:

Accuracy: 95.1% (39/41 correct predictions)
Precision: 96.3% (26 true positives, only 1 false positive)
Recall: 96.3% (caught 26/27 tumors, missed only 1)
Specificity: 92.9% (correctly identified 13/14 healthy brains)

Training metrics were noisy due to the small dataset, 
but validation metrics stabilized after ~40 epochs. 
The key insight was that the model needed the full 100 epochs to learn instead of The early stopping which 
ended training around 10-20 epochs.

Q: How did you interpret your confusion matrix and what insights did it provide?  
A: Final confusion matrix on validation set (41 samples):

True Positives: 26 | False Positives: 1
True Negatives: 13 | False Negatives: 1

The model achieved excellent balance. Only 1 false alarm (healthy brain classified as tumor) 
and only 1 missed tumor. Initially, the model struggled with healthy brains (10 false positives), 
but extended training dramatically improved specificity from 29% to 93%. 
The prediction probabilities also became more confident, spanning the full 0.0-1.0 range instead of clustering around 0.5.

Q: What was your model's performance on the test set compared to validation set?  
A:  The results between the two sets were pretty similar.
                  TRAINING      VALIDATION
		Accuracy  ~92%		    95.1%
		Recall    ~90%			96.3% 
		Precision ~93%	    	96.6%
		Loss      ~0.3			~0.3


### 🔄 6. Transfer Learning Comparison (optional)

Q: Which pre-trained model did you use for transfer learning (MobileNetV2, ResNet50, etc.)?  
A:  N/A

Q: Did you freeze the base model layers or allow fine-tuning? Why?  
A:  N/A

Q: How did transfer learning performance compare to your custom CNN?  
A:  N/A


### 🔍 7. Error Analysis & Model Insights

Q: What types of images does your model most commonly misclassify?  
A: With only 2 misclassifications out of 41 validation samples (1 false positive, 1 false negative), 
   the error rate is very low. The model's predictions now span confidently from 0.0 to 1.0, with a 
   median of 0.68, indicating it has learned meaningful distinguishing features. Early in training, 
   the model showed bias toward predicting tumors (10 false positives), but extended training corrected this.

Q: How did you analyze and visualize your model's mistakes?  
A: I used confusion matrices, precision-recall curves, and tracked metrics across all 100 epochs. The training 
   history graphs revealed that validation precision improved from ~75% at epoch 40 to ~90% at epoch 100, showing 
   the model continued learning throughout. I also analyzed prediction probability distributions (min, max, mean, 
   median) to ensure the model was making confident predictions rather than uncertain ones near the 0.5 threshold.

Q: What improvements would you make based on your error analysis?  
A: Given the 95% accuracy achieved, the main improvements would be:

   Collect more data - 205 images is very small; 500-1000+ would improve the models generalization.
   Analyze the 2 misclassified images - Examine what makes them difficult to understand edge cases.
   Class weights - Could potentially eliminate the remaining errors, though diminishing returns at 95% accuracy.

