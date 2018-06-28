gitlab_ci:
	pip install pipenv && \
	pipenv install --dev && \
	pipenv run pytest -svv --cov sleepmind

build_w2v:
	# download pre-trained w2v model
	wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" && \
	gunzip GoogleNews-vectors-negative300.bin.gz

test:
	pipenv run pytest -svv --cov sleepmind
