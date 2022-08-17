# COBE

## Libraries:

python==3.8.10

transformers==4.1.1

torch==1.10.0

## Run:

You can run the code for the task of cross-domain sentiment analysis for FDU-MTL dataset:
```
python run_COBE.py
```

You can change the parameters of --train_domains and --test_domains for different tasks.

As for cross-domain Amazon dataset, you need to change ```SentProcessor()``` to ```SentProcessor2()```.

After running, a result file can be found in the output directory (named test_results.txt).

## Cite:
