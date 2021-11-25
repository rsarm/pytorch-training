# Text summarization with T5 on XSum

We are going to fine-tune [T5 implemented by HuggingFace](https://huggingface.co/t5-small) for text summarization on the [Extreme Summarization (XSum)](https://huggingface.co/datasets/xsum) dataset.
The data if composed by news articles and the corresponding summaries.

This notebooks is based on the script [run_summarization_no_trainer.py](https://github.com/huggingface/transformers/blob/v4.12.5/examples/pytorch/summarization/run_summarization_no_trainer.py) from HuggingFace.

More info:
* [T5 on HuggingFace docs](https://huggingface.co/transformers/model_doc/t5.html)
