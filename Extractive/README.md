# Extractive Experiments
We follow [previous work](https://github.com/xiangyue9607/QVE) to generate synthetic QA
 pairs and evaluate downstream performance after augmentation.

## Data Preparation
The experiments are focus on semi-supervised domain adaptation setting.
We use "SQuAD" as the source dataset and "NaturalQuestionsShort"
"TriviaQA-web" as target datasets. All the datasets can be downloaded
from [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019).

We use the original dev set as the test set and sample a limited number
(by default: 1000) of QA pairs from the training as the dev set. Since
there is no test set available for each dataset, we use the original dev
set as the test set and sample 1,000 QA pairs from each target domain as
the dev set.
```shell script
$ ./download_and_process.sh
```

## QG
The QG model is trained on the source domain and then finetuned on the
limited target dev set. And then we use the finetuned QG model to
generate synthetic questions on all the target contexts. We finally
convert the synthetic questions into the QA data format.
```shell script
$ ./run_qg.sh
```

## Selection
See RL_Selection section.

## QA
The QA model is pretrained on the source and then finetuned on the target
 synthetic and target dev sample. We use EM and F1 on the test set to
 evaluate the performance of the QA model after augmented training.
```shell script
$ ./run_qa_baseline.sh
```

