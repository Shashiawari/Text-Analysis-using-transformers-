# Text-Analysis-using-transformers-


This project implements **sentiment classification** using **transformers** (BERT) to analyze text sentiment. It preprocesses text data, fine-tunes a transformer model, and evaluates classification performance.

## 🚀 Features
- **Data Preprocessing**: Cleans and tokenizes text.
- **Transformer-Based Model**: Fine-tunes **BERT** for sentiment classification.
- **Performance Evaluation**: Computes accuracy, precision, recall, and **confusion matrix**.
- **Visualization**: Plots **training loss**, **validation accuracy**, and **confusion matrix**.
- **Real-Time Inference**: Predicts sentiment for new text inputs.

## 📂 Dataset
- Uses a dataset of sentiment-labeled text (from `em.csv`).
- Sentiment labels range from **-5 to -1** (strongly negative to slightly negative).

## 🔧 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/shashiawari/Text-Analysis-using-transformers.git
Install dependencies:
bash
Copy
Edit
pip install transformers[torch] pandas datasets scikit-learn seaborn matplotlib
Run the training script:
bash
Copy
Edit
python train.py
📊 Model Evaluation
Accuracy: 35% on the test set.
Confusion Matrix:
Shows the distribution of correct and incorrect classifications.
Classification Report:
Precision, recall, and F1-score for each sentiment category.
🖥️ Usage
To predict sentiment from text:

python
Copy
Edit
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def get_prediction(text, model):
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = model(**encoding)
    probs = torch.nn.Softmax(dim=1)(outputs.logits)
    label = torch.argmax(probs).item()
    return label

prediction = get_prediction("I love this!", model)
print("Predicted Sentiment:", prediction)
📌 Next Steps
Improve accuracy with hyperparameter tuning.
Expand dataset with balanced sentiment classes.
Deploy model using FastAPI or Flask.
🤝 Contributing
Feel free to submit issues or pull requests! 🚀




