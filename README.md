# Fine-Tune GenAI Model for DialogSum

This project fine-tunes a Generative AI model for dialogue summarization using the DialogSum dataset.

## Setup Instructions

### Prerequisites

Ensure you are using an `ml.m5.2xlarge` instance (8 vCPUs, 32 GiB RAM) for optimal performance.

### Installation

1. Upgrade `pip`:
   ```sh
   pip install --upgrade pip
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Training the Model

Run the Jupyter Notebook to execute the fine-tuning process:

```sh
jupyter notebook Fine_tune_genai_mode_dialogsum.ipynb
```

## Fine-Tuning Approaches

We perform two types of fine-tuning:

1. **Full Fine-Tuning:** The entire model is fine-tuned on the DialogSum dataset, updating all parameters to optimize performance.
2. **Parameter-Efficient Fine-Tuning (PEFT):** Instead of updating the entire model, PEFT techniques such as LoRA are used to fine-tune a smaller subset of parameters, making it more efficient in terms of computation and memory.

## Model Evaluation

We use the **ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric** to evaluate and compare the performance of different fine-tuning approaches. ROUGE measures the overlap between generated summaries and reference summaries, providing insights into the model's summarization quality.

## Dependencies

The project relies on:

- TensorFlow==2.12.0 & Keras==2.12.0
- PyTorch==1.13.1 & TorchData==0.6.0
- Hugging Face Transformers==4.27.2, Datasets==2.17.0, and PEFT==0.3.0
- Evaluate==0.4.0 & ROUGE Score==0.1.2
- TRL (installed from source)


