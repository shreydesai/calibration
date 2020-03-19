# Calibration of Pre-trained Transformers

Code and datasets for our preprint [Calibration of Pre-trained Transformers](https://arxiv.org/abs/2003.07892). If you found this project helpful, please consider citing our paper:

```bibtex
@article{desai-durrett-2020-calibration,
  author={Desai, Shrey and Durrett, Greg},
  title={{Calibration of Pre-trained Transformers}},
  year={2020},
  journal={arXiv preprint arXiv:1907.11692},
}
```

## Overview

Posterior calibration is a measure of how aligned a model's posterior probabilities are with empirical likelihoods. For example, a perfectly calibrated model that outputs 0.8 probability on 100 samples should get 80% of the samples correct. In this work, we analyze the calibration of two pre-trained Transformers (BERT and RoBERTa) on three tasks: natural language inference, paraphrase detection, and commonsense reasoning.

For natural language inference, we use [Stanford Natural Language Inference](https://nlp.stanford.edu/projects/snli/) (SNLI) and [Multi-Genre Natural Language Inference](https://www.nyu.edu/projects/bowman/multinli/) (MNLI). For paraphrase detection, we use [Quora Question Pairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) (QQP) and [TwitterPPDB](https://languagenet.github.io/) (TPPDB). And, for commonsense reasoning, we use [Situations with Adversarial Generations](https://rowanzellers.com/swag/) (SWAG) and [HellaSWAG](https://rowanzellers.com/hellaswag/) (HSWAG).

To measure calibration error, we chiefly use expected calibration error (ECE), where ECE = 0 indicates perfect calibration. When used in-domain, pre-trained models ([BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692)) are generally **much more calibrated** than non-pre-trained models ([DA](https://arxiv.org/abs/1606.01933), [ESIM](https://arxiv.org/abs/1609.06038)). For example, on SWAG:

| Model   | Accuracy |  ECE |
|---------|:--------:|:----:|
| DA      |   46.80  | 5.98 |
| ESIM    |   52.09  | 7.01 |
| BERT    |   79.40  | 2.49 |
| RoBERTa |   82.45  | 1.76 |

To bring down calibration error, we experiment with two strategies. First, **temperature scaling** (TS; dividing non-normalized logits by scalar T) almost always brings ECE below 1. Below, we show in-domain results with and without temperature scaling:

| Model         | SNLI | QQP  | SWAG |
|---------------|:----:|:----:|:----:|
| RoBERTa       | 1.93 | 2.33 | 1.76 |
| RoBERTa (+TS) | 0.84 | 0.88 | 0.76 |

Second, deliberately inducing uncertainty via **label smoothing** (LS) helps calibrate posteriors out-of-domain. MLE training encourages models to be over-confident, which is typically unwarranted out-of-domain, where models should be uncertain. We show out-of-domain results with and without label smoothing:

| Model       | MNLI | TPPDB |  HSWAG |
|--------------|:----:|:-----:|:-----:|
| RoBERTa-MLE  | 3.62 |  9.55 | 11.93 |
| RoBERTa-LS   | 4.50 |  8.91 | 2.14  |

Please see [our paper](https://arxiv.org/abs/2003.07892) for the complete set of experiments and results!

## Instructions

### Requirements

This repository has the following requirements:

- `numpy==1.18.1`
- `scikit-learn==0.22.1`
- `torch==1.2.0`
- `tqdm==4.42.1`
- `transformers==2.4.1`

Use the following instructions to set up the dependencies:

```bash
$ virtualenv -p python3.6 venv
$ pip install -r requirements.txt
```

### Obtaining Datasets

Because many of these tasks are either included in the GLUE benchmark or commonly used to evaluate pre-trained models, the test sets are blind. Therefore, we split the development set in half to obtain a non-blind, held-out test set. The dataset splits are shown below, and additionally, you may [download the exact train/dev/test datasets](https://drive.google.com/file/d/1ro3Q7019AtGYSG76KeZQSq35lBi7lU3v/view?usp=sharing) used in our experiments. Use `tar -zxf calibration_data.tar.gz` to unpack the archive, and place it in the root directory.

| Dataset |  Train  |   Dev  |  Test  |
|---------|:-------:|:------:|:------:|
| SNLI    | 549,368 |  4,922 |  4,923 |
| MNLI    | 392,702 |  4,908 |  4,907 |
| QQP     | 363,871 | 20,216 | 20,217 |
| TPPDB   |  46,667 |  5,060 |  5,060 |
| SWAG    |  73,547 | 10,004 | 10,004 |
| HSWAG   |  39,905 |  5,021 |  5,021 |

### Fine-tuning Models

Coming soon!

### Evaluating Calibration

Coming soon!
