PY=./.venv/bin/python3
PIP=./.venv/bin/pip3

DATASET=./dataset

all: reqs $(DATASET)
	$(PY) src/main.py

reqs: requirements.txt
	$(PIP) install -r requirements.txt

$(DATASET):
	mkdir -p $(DATASET)
	curl -L -o $(DATASET)/poland-vehicle-license-plate-dataset.zip https://www.kaggle.com/api/v1/datasets/download/piotrstefaskiue/poland-vehicle-license-plate-dataset
	unzip $(DATASET)/poland-vehicle-license-plate-dataset.zip -d $(DATASET)
	rm -rf $(DATASET)/poland-vehicle-license-plate-dataset.zip
