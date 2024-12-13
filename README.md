# Code Generation with Salesforce CodeGen

This project demonstrates how to use the Salesforce CodeGen model (`codegen2-1B`) for Python code generation. 
It includes steps for preprocessing, generation, and evaluation using metrics like BLEU score and perplexity.

## Features
- **Model**: Salesforce CodeGen (`codegen2-1B`)
- **Metrics**: BLEU score for similarity and perplexity for model evaluation.
- **Flexibility**: Easily extendable to other tasks with different prompts.

## Requirements
The following dependencies are required:
- `transformers==4.33.3`
- `torch==2.0.1`
- `nltk==3.8.1`

## Installation
1. Clone the repository or copy the code files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
