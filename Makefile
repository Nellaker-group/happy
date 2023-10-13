environment_cu117:
	pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
	pip install torch_geometric==2.3.1
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install -e .
.PHONY: install

environment_cpu:
	pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
	pip install torch_geometric==2.3.1
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
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

