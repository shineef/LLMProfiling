# LLM Profiling and Fine-Tuning with Limited Neighbor Information for Node Classification on Text-Attributed Graphs

This repository contains the script files for the [ArxivLlama](https://huggingface.co/xinyifang/ArxivLlama) & [ProductsLlama](https://huggingface.co/xinyifang/ProductsLlama) model in our paper. Below are the instructions for those files:

### Get the data

For ogbn-arxiv and ogbn-products datasets, follow files in <mark>dataget</mark> folder.

* Run <mark>ogbnget.py</mark> to download the files to prepare and split the datasets.

* Run <mark>originalget.sh</mark> to download the original text version of the two text-attributed graphs.

### Before the construction

Please run those scripts before constructing the models

* <mark>arxivsplit.py</mark> is for splitting train, validation and test data.

* <mark>idtrans.py</mark> and <mark>pid.py</mark> are for transferring or adding indexes to the original text data.

<mark>unfuned.py</mark> is a comparison of node classification without using our model, but only using a language model for the task.

### Node profiles generation and construction

We use 5 different language models to generate different profiles. Ollama pre-installation is required in this process. The specific installation process depends on different systems. Please follow the official [installation instructions](https://ollama.com/download) on the ollama website.

Run <mark>profileextract.py</mark> to generate profiles of the original data.

Run <mark>structuremaker.py</mark> to construct the prompts that we can use for fine-tuning the backbone model.

### Finetuning

The <mark>finetune.py</mark> file is for finetuning backbone model. We used Unsloth to accelerate this process, so that Unsloth pre-installation is required. 

* Please follow this [official guide](https://github.com/unslothai/unsloth) for installation. According to our experiments, a single A5000 is sufficient for finetuning in this process.

* Please make sure there are no other .json files in the same folder except those generated from <mark>structuremaker.py</mark>.

### Node classification task

The <mark>arxivclassifier.py</mark> and <mark>productclassifier.py</mark> files are for the node classification task.



