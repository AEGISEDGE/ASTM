# Code for "Improving neural topic modeling via Sinkhorn divergence"

## Requirements:
	python3.8	

	pytorch1.7 + CUDA11.0

	geomloss

	pykeops

	numpy

	matplotlib

	prefect_generator



## Note:

Directory description:

'data/20news': default dataset path.

'embedding_dir': word embedding file path. 

'sav':  checkpoint file path. 

'embedding_dir/word2vec_glove.6B.100d.txt.bin': 100-dimension Glove word embedding dictionary file.

'corpus_obj.bin': Preprocessed 20NewsGroup binary corpus for rapid load and training.


## Run:

Download the word embedding file form the link given in "embedding_dir/download_link.md". Then u can run the model by:
	
	python run.py --topics 50 --batch-size=64 --lr=1e-4 --coel=0.1 --coea=5.0 --topk=10

## Cite:

if u find this code useful, plz kindly cite our paper:

	@article{LIU2022102864,
	title = {Improving neural topic modeling via Sinkhorn divergence},
	journal = {Information Processing & Management},
	volume = {59},
	number = {3},
	pages = {102864},
	year = {2022},
	issn = {0306-4573},
	doi = {https://doi.org/10.1016/j.ipm.2021.102864},
	url = {https://www.sciencedirect.com/science/article/pii/S0306457321003356},
	author = {Luyang Liu and Heyan Huang and Yang Gao and Yongfeng Zhang},
	keywords = {Deep learning, Topic model, Sinkhorn divergence, Auto-encoder}
	}

