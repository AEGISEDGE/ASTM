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

'data/20news': default dataset path.

'embedding_dir': word embedding file path. 

'sav':  checkpoint file path. 

'embedding_dir/word2vec_glove.6B.100d.txt.bin': 100-dimension Glove word embedding dictionary file.

'corpus_obj.bin': Preprocessed 20NewsGroup binary corpus for rapid load and training.


Run:
	
	python run.py --topics 50 --batch-size=64 --lr=1e-4 --coel=0.1 --coea=5.0 --topk=10


