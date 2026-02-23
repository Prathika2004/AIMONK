
##Multi-label Image Classification (Aimonk Assignment)

This project implements a deep learning solution for a multi-label image classification task where images can have multiple attributes simultaneously (Attr1-Attr4).

🚀 Key Challenges & Solutions
1. Handling Missing Information (The "NA" Problem)

Challenge: Some images have "NA" for certain attributes. Standard loss functions would fail or learn incorrect patterns from these.
Solution: I implemented a Masked Binary Cross-Entropy (BCE) Loss.

I mapped "NA" values to a placeholder (-1).

During training, the loss function creates a "binary mask" that ignores the gradients for any attribute marked as -1. This ensures the model only learns from confirmed data without discarding the entire image.

2. Handling Data Imbalance (Skewness)

Challenge: Some attributes appear much less frequently than others, leading to a biased model.
Solution: I implemented Positional Weights in the loss function. I calculated the ratio of negative-to-positive samples for each attribute and passed these as pos_weight to the optimizer. This forces the model to prioritize learning the rare attributes.

3. Transfer Learning & Architecture

Backbone: ResNet50 (Pre-trained on ImageNet).

Fine-tuning: I replaced the final fully connected layer to output 4 neurons (one for each attribute) and trained with a low learning rate to preserve the pre-trained feature extraction capabilities.

🛠️ Technical Implementation
Data Preprocessing

Cleaning: Handled string inconsistencies (e.g., converting "1mobilenet" to "1").

Syncing: Implemented a verification step to ensure the dataframe only contains images that physically exist on the disk.

Augmentation: Applied Resizing (224x224), Normalization, and Random Horizontal Flips to improve generalization.

Deliverables included:

Training Code: Script to fine-tune the model and save weights.

Loss Curve: A plot titled Aimonk_multilabel_problem showing training_loss vs iteration_number.

Inference Script: A function to take a raw image and print the list of detected attributes.

📈 Results

The model was trained for 20 epochs, achieving a significant reduction in loss:

Starting Loss: ~0.60

Final Loss: ~0.04


Directly answers the prompt: It uses the same words as the assignment (NA, Skewed, Imbalance).

Shows Mathematical Maturity: Mentioning "Masked Gradients" and "Positional Weights" shows you aren't just a beginner.

Professional Formatting: Using headers and bullet points makes it easy for a recruiter to skim.
