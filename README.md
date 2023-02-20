# Code for "Improving neural topic modeling via Sinkhorn divergence"

--- 

## Requirements:
	python3.8	

	pytorch1.7 + CUDA11.0

	geomloss

	pykeops

	numpy

	matplotlib

	prefect_generator

---


## Note:

Directory description:

	'data': default dataset path.
	'embedding_dir': word embedding file path. 
	'sav':  checkpoint file path. 

Necessary files:

	'corpus_obj.bin': Preprocessed 20NewsGroup binary corpus.
	'embedding_dir/word2vec_glove.6B.100d.txt.bin': 100-dimension Glove word embedding dictionary file.

Run:
	
	python run.py 	--topics <the number of topics>
					--n-hidden <the number of hidden units>
				  	--lr <learning rate>
					--dropout <probabilty of dropout layer>
					--topicembedsize <the dimension number of topic embeddings and word embeddings >
					--batch-size <batch size for training>
					--data-path <ur own datapath>
					--coel <coefficient for Sinkhorn divergence term>
					--coea <coefficient for STDR>
					--topk <the number of top words to extract by STDR>
