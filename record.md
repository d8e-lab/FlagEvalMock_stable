hf-causal  
**model**: pretrained=/mnt/40_store/SiYuan/llama2-checkpoints-plus-longer_checkpoint-27000_data1103_trick1101-merge 
limit: None, provide_description: False, num_fewshot: 0, batch_size: None  
![](ckpt27000-nodrop&mmlu.png)

|Task|Version|Metric|Value |   |Stderr|
|----|------:|------|-----:|---|-----:|
|drop|      1|em    |0.0131|±  |0.0012|
|    |       |f1    |0.0970|±  |0.0019|

----

hf-causal-experimental  
**model**: pretrained=/mnt/SFT_store/wyh/outputs/checkpoint9750_data1114_trick1101/ 
limit: None
, provide_description: False, num_fewshot: 0, batch_size: None                                                
|     Task     |Version|  Metric   | Value |   |Stderr|
|--------------|------:|-----------|------:|---|-----:|
|gsm8k         |      0|acc        | 0.0000|±  |0.0000|
|truthfulqa_gen|      1|bleurt_max |-0.5215|±  |0.0168|
|              |       |bleurt_acc | 0.4541|±  |0.0174|
|              |       |bleurt_diff|-0.0087|±  |0.0207|
|              |       |bleu_max   |26.4748|±  |0.8041|
|              |       |bleu_acc   | 0.4088|±  |0.0172|
|              |       |bleu_diff  |-1.0620|±  |0.8577|
|              |       |rouge1_max |53.8562|±  |0.8797|
|              |       |rouge1_acc | 0.4149|±  |0.0172|
|              |       |rouge1_diff|-0.3707|±  |1.0345|
|              |       |rouge2_max |38.6756|±  |1.0510|
|              |       |rouge2_acc | 0.3488|±  |0.0167|
|              |       |rouge2_diff|-1.5606|±  |1.1918|
|              |       |rougeL_max |50.9375|±  |0.9125|
|              |       |rougeL_acc | 0.4125|±  |0.0172|
|              |       |rougeL_diff|-0.4182|±  |1.0439|
|hellaswag     |      0|acc        | 0.5621|±  |0.0050|
|              |       |acc_norm   | 0.7534|±  |0.0043|
|truthfulqa_mc |      1|mc1        | 0.3133|±  |0.0162|
|              |       |mc2        | 0.4529|±  |0.0152|
|winogrande    |      0|acc        | 0.6819|±  |0.0131|
|drop          |      1|em         |0.1036 |±  |0.0031|
|              |       |f1         |0.1610 |±  |0.0032|


----

hf-causal   
**model** pretrained=/mnt/40_store/SiYuan/chat_sftv3-9750
|Task|Version|Metric|Value |   |Stderr|
|----|------:|------|-----:|---|-----:|
|drop|      1|em    |0.0026|±  |0.0005|
|    |       |f1    |0.0661|±  |0.0013|
|hellaswag|      0|acc     |0.5706|±  |0.0049|
|         |       |acc_norm|0.7707|±  |0.0042|
|truthfulqa_gen|      1|bleurt_max |-0.4784|±  |0.0161|
|              |       |bleurt_acc | 0.5067|±  |0.0175|
|              |       |bleurt_diff| 0.1014|±  |0.0245|
|              |       |bleu_max   |30.0117|±  |0.7727|
|              |       |bleu_acc   | 0.4455|±  |0.0174|
|              |       |bleu_diff  | 2.7287|±  |0.9865|
|              |       |rouge1_max |59.0525|±  |0.8751|
|              |       |rouge1_acc | 0.4529|±  |0.0174|
|              |       |rouge1_diff| 6.8284|±  |1.3546|
|              |       |rouge2_max |45.5763|±  |1.0659|
|              |       |rouge2_acc | 0.4051|±  |0.0172|
|              |       |rouge2_diff| 6.2794|±  |1.4958|
|              |       |rougeL_max |56.3705|±  |0.9149|
|              |       |rougeL_acc | 0.4443|±  |0.0174|
|              |       |rougeL_diff| 6.8736|±  |1.3631|
|truthfulqa_mc |      1|mc1        | 0.3072|±  |0.0162|
|              |       |mc2        | 0.4517|±  |0.0149|
|winogrande|      0|acc   |0.7316|±  |0.0125|