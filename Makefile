help:
	@cat Makefile

DOCKER=GPU=0 nvidia-docker
BACKEND=tensorflow
SRC=`pwd`

build:
	docker build -t nst -f Dockerfile .

bash: build
	$(DOCKER) run -it -v $(SRC):/src/nst -p 8502:8501 --env KERAS_BACKEND=$(BACKEND) nst bash

gpu-app: build
	$(DOCKER) run -it -v $(SRC):/src/nst -p 8502:8501 --env KERAS_BACKEND=$(BACKEND) nst streamlit run app.py

gpu-test-run: build
	$(DOCKER) run -it -v $(SRC):/src/nst -p 8502:8501 --env KERAS_BACKEND=$(BACKEND) nst pytest nst/tests.py

cpu-app:
	streamlit run app.py

cpu-test-run:
	pytest nst/tests.py
