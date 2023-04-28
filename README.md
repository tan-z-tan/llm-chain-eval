## LLM Chain Eval
LLM Chain Eval is a versatile and powerful tool designed to adjust parameters and evaluate Language Models (LMs) such as GPT-3.5-turbo, GPT-4 to obtain the desired output based on various input parameters. This repository allows you to experiment with different parameters, analyze input-output behavior, and assess the overall performance of the model.

## Features
Currently supports only Question-Answering (QA) input-output scenarios.
Aims to become a more general-purpose tool for evaluating LMs across different tasks and applications.
Open to contributions from the community to expand its capabilities.

### LLM Model
- OpenAI gpt-3.5-turbo
- OpenAI gpt-4
- google/flan-t5-large (HuggingFaceHub)
- google/flan-t5-xl (HuggingFaceHub)
- databricks/dolly-v2-3b (TBD)

### Embedding
- OpenAIEmbeddings
- HuggingFaceEmbeddings
- Flan-t5 (TBD)

### Parameters
- Chunk Size
- Chunk Overlap
- Chain type

### Prompt templates
- You can modify prompt for stuff QA.
- Match template to see if the answer is correct.

## Setup
1. Clone the repository:

```
git clone https://github.com/tan-z-tan/llm-chain-eval.git
cd llm-chain-eval
```

```
pip install -r requirements.txt
```

Put your OPENAI_API_KEY `XXX`.
```
echo OPENAI_API_KEY=XXX > .env
```

Put your HUGGINGFACEHUB_API_TOKE `YYY` if you want to use HuggingFaceHub.
```
echo HUGGINGFACEHUB_API_TOKEN=YYY >> .env
```

## Run
```shell
streamlit run app.py 
```

- Upload source file (see sample_text.txt)
- Upload validation file (see sample_validation.txt)

<img src="app.jpg"/>

## Contributing
We welcome contributions from the community to help make LLM Chain Eval a more powerful and versatile tool. If you have any ideas, suggestions, or improvements, please feel free to submit a pull request or open an issue. We appreciate your support in making this tool better for everyone!

## License
This project is licensed under the MIT License.
