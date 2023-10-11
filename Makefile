environment_cu113:
	pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
	pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install -e .
.PHONY: install

environment_cpu:
	pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
	pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install -e .
.PHONY: install

setup:
	pip install -e .
.PHONY: install

test:
	pytest
.PHONY: install

fmt:
	black happy projects/placenta analysis qupath
.PHONY: install

