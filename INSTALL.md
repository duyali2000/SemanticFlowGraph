## Fine-tuning 
* Evaluate and fine-tune the pre-trained model with D-, ARC-, and ARCL-encoding on commits granularity.
```
sh run_commits_D.sh & dataset & cuda number
sh run_commits_ARC.sh & dataset & cuda number
sh run_commits_ARCL.sh & dataset & cuda number
```
* Evaluate and fine-tune the pre-trained model with D-, ARC- and ARCL-encoding on file granularity.
```
sh run_files_D.sh & dataset & cuda number
sh run_files_ARC.sh & dataset & cuda number
sh run_files_ARCL.sh & dataset & cuda number
```
* Evaluate and fine-tune the pre-trained model with D-, ARC-, and ARCL-encoding on hunks granularity.
```
sh run_hunks_D.sh & dataset & cuda number
sh run_hunks_ARC.sh & dataset & cuda number
sh run_hunks_ARCL.sh & dataset & cuda number
```
Evaluation results are saved to `retrieval_result.csv` in the `results` directory.