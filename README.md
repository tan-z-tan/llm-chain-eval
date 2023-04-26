## LLM Chain Eval
LLM Chain Eval is a versatile and powerful tool designed to adjust parameters and evaluate Language Models (LMs) such as GPT-3.5-turbo, GPT-4 to obtain the desired output based on various input parameters. This repository allows you to experiment with different parameters, analyze input-output behavior, and assess the overall performance of the model.

## Features
Currently supports only Question-Answering (QA) input-output scenarios.
Aims to become a more general-purpose tool for evaluating LMs across different tasks and applications.
Open to contributions from the community to expand its capabilities.

## Setup
1. Clone the repository:

```
git clone https://github.com/tan-z-tan/llm-chain-eval.git
cd llm-chain-eval
```

Use poetry to install libraries.
```
poetry install
```
or use pip
```
pip install -r requirements.txt
```

## Run
```
poetry run streamlit run app.py 
```
or
```
streamlit run app.py 
```

## Contributing
We welcome contributions from the community to help make LLM Chain Eval a more powerful and versatile tool. If you have any ideas, suggestions, or improvements, please feel free to submit a pull request or open an issue. We appreciate your support in making this tool better for everyone!

## License
This project is licensed under the MIT License.
