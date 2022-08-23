# COBE

## Libraries:

python==3.8.10

transformers==4.1.1

pytorch==1.10.0

## Run:

You can run the code for the task of cross-domain sentiment analysis for FDU-MTL dataset:
```
python run_COBE.py
```

You can change the parameters of --train_domains and --test_domains for different tasks.

As for cross-domain Amazon dataset, you need to change ```SentProcessor()``` to ```SentProcessor2()```.

After running, a result file can be found in the output directory (named test_results.txt).

## Cite:

@article{luo2022mere,

  title={Mere Contrastive Learning for Cross-Domain Sentiment Analysis},
  
  author={Luo, Yun and Guo, Fang and Liu, Zihan and Zhang, Yue},
  
  journal={arXiv preprint arXiv:2208.08678},
  
  year={2022}
  
}
