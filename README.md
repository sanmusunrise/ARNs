# Sequence-to-Nuggets: Nested Entity Mention Detection via Anchor-Region Networks

This is the source code for paper "Sequence-to-Nuggets: Nested Entity Mention Detection via Anchor-Region Networks" in ACL2019.

## Requirements

* Pytorch >= 0.4.0
* tabulate == 0.8.3

## Usage
First, please unzip the datasets in "data/"

* tar zxvf ACE2005.tgz
* tar zxvf GENIA.tgz
* tar zxvf KBP2017.tgz

Then enter "src/workspace_*" dir, run the program like

* python main_seq2nugget.py

Hyperparameters in our paper are saved in "main_seq2nugget.py" files.

## Additional Notes on KBP2017 Evaluation
KBP2017 contains forum data and the original annoatations marked the author fields of each post as either "PER" or "ORG". This can result in an undervalued recall rate if we did not take this into consideration because the data used in our model removed all author fields. 

Following all previous works, we annotate all post author field in original data as "PER" in "kbp2017_evaluator/kbp2017_test_eval_files/authors.dat". Then for correct evaluation, we need to combine the "authors.dat" with the system output, and then use the official evalution toolkit to obtain the accurate evaluation result.

For this, we should first install the offical evaluation toolkit with command
* pip install git+https://github.com/wikilinks/neleval

And then transform our system output into the required format (also combine it with the author fields) by entering "kbp2017_evaluator" dir and then use the command:
* python generate_eval_format.py [test_result_file] sys_output_for_eval.dat

Finally, we obtain the offical evalution result with command
* neleval evaluate -m strong_typed_mention_match -f tab -g kbp2017_test_golden.dat sys_output_for_eval.dat

The "test_result_file" can be found in the Model dir in workspace. We also  provide a sample test_result in "kbp2017_test_eval_files/sample_sys_test_output.dat" which is produced by the model reported in our paper.

## Citation
Please cite:
* Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun. *Sequence-to-Nuggets: Nested Entity Mention Detection via Anchor-Region Networks*. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

```
@InProceedings{lin-Etal:2019:ACL2019sequence,
  author    = {Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le},
  title     = {Sequence-to-Nuggets: Nested Entity Mention Detection via Anchor-Region Networks},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2019},
  publisher = {Association for Computational Linguistics}
}
```

## Contact
If you have any question or want to request for the used datasets (ACE2005, GENIA and KBP2017 if you have the license), please contact me by
* hongyu2016@iscas.ac.cn
