# PASTA
This is the repository for the resources of paper "**Pa**rameter-efficient Tuning with **S**pecial **T**oken **A**daptation".

## Abstract
Parameter-efficient tuning aims at updating only a small subset of parameters when adapting a pretrained model to downstream tasks. In this work, we introduce **PASTA**, in which we only modify the special token representations (e.g., *[SEP]* and *[CLS]* in BERT) before the self-attention module at each layer in Transformer-based models. PASTA achieves comparable performance to fine-tuning in natural language understanding tasks including text classification and NER with up to only 0.029% of total parameters trained. Our work not only provides a simple yet effective way of parameter-efficient tuning, which has a wide range of practical applications when deploying finetuned models for multiple tasks, but also demonstrates the pivotal role of special tokens in pretrained language models.

<img width="912" alt="image" src="https://user-images.githubusercontent.com/79353358/194789309-f4991392-5d9b-4786-8463-636797028ca3.png">
