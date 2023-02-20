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

## Cite

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
	keywords = {Deep learning, Topic model, Sinkhorn divergence, Auto-encoder},
	abstract = {Textual data have been a major form to convey internet users’ content. How to effectively and efficiently discover latent topics among them has essential theoretical and practical value. Recently, neural topic models(NTMs), especially Variational Auto-encoder-based NTMs, proved to be a successful approach for mining meaningful and interpretable topics. However, they usually suffer from two major issues:(1)Posterior collapse: KL divergence will rapidly reach zeros resulting in low-quality representation in latent distribution; (2)Unconstrained topic generative models: Topic generative models are always unconstrained, which potentially leads to discovering redundant topics. To address these issues, we propose Autoencoding Sinkhorn Topic Model based on Sinkhorn Auto-encoder(SAE) and Sinkhorn divergence. SAE utilizes Sinkhorn divergence rather than problematic KL divergence to optimize the difference between posterior and prior, which is free of posterior collapse. Then, to reduce topic redundancy, Sinkhorn Topic Diversity Regularization(STDR) is presented. STDR leverages the proposed Salient Topic Layer and Sinkhorn divergence for measuring distance between salient topic features and serves as a penalty term in loss function facilitating discovering diversified topics in training. Several experiments have been conducted on 2 popular datasets to verify our contribution. Experiment results demonstrate the effectiveness of the proposed model.}
	}

