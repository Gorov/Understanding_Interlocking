# Understanding Interlocking
This repo contains the PyTorch implementation of [Understanding Interlocking Dynamics of Cooperative Rationalization](https://arxiv.org/abs/2110.13880). 

The original [beer review dataset](http://snap.stanford.edu/data/web-BeerAdvocate.html) has been removed by the dataset’s original author, at the request of the data owner, BeerAdvocate.  To respect the wishes and legal rights of the data owner, we do not include the data in our repo.  

If you are interested in beer review, please first obtain the dataset from the original authors who released the dataset; then forward us the agreement email from the authors. We will then be happy to provide our data in our processed format to whoever is granted rights to the data.  


```
@article{yu2021understanding,
  title={Understanding Interlocking Dynamics of Cooperative Rationalization},
  author={Yu, Mo and Zhang, Yang and Chang, Shiyu and Jaakkola, Tommi},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Getting Started
Directly running the provided main jupyter-notebook `run_beer_arc2_sentence_level_neurips21.ipynb` should reproduce the ~70 F1 on the Aroma aspect of Beer Review.

We also provide example scripts for running our beer-biased setting (`run_beer_arc2_sentence_level_toy_biased_aspect2_neurips21.ipynb`, which follows the more strict setting, i.e., Table 6 in our Appendix) and the movie review setting (`run_movie_arc2_token_level_neurips21.ipynb`). The beer-skew setting can be enabled by setting the `switch_epoch` variable to non-zero intergers in our main ipynb.


**Tested environment:**
Python 3.7.6, pytorch=1.4.0=cuda101py37h02f0884_0, torchtext==0.4.0, cudatoolkit=10.1.243=h6bb024c_0, and cudnn=7.6.5=cuda10.1_0.



## Final Words
That's all for now and hope this repo is useful to your research.  For any questions, please create an issue and we will get back to you as soon as possible.

