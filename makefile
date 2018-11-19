build_w2v:
	# download pre-trained w2v model
	wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" && \
	gunzip GoogleNews-vectors-negative300.bin.gz


build:
	pip install -e .


test:
	pytest -svv --cov sleepmind
