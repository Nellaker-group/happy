.PHONY: environment_cpu environment_cu118 setup test fmt

environment_cpu:
	python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
	python -m pip install torch_geometric==2.4.0
	python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
	python -m pip install -r requirements.txt
	python -m pip install datashader==0.15.2
	python -m pip install "holoviews[recommended]"
	python -m pip install -e .

environment_cu117:
	python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
	python -m pip install torch_geometric==2.3.1
	python -m pip install pyg_lib==0.2.0 torch_scatter torch_sparse==0.6.17 torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
	python -m pip install -r requirements.txt
	python -m pip install pyvips==2.2.1
	python -m pip install datashader==0.14.4
	python -m pip install "holoviews[recommended]"
	python -m pip install -e .

environment_cu118:
	python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
	python -m pip install torch_geometric==2.4.0
	python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
	python -m pip install -r requirements.txt
	python -m pip install datashader==0.15.2
	python -m pip install "holoviews[recommended]"
	python -m pip install -e .

environment_cu121:
	python -m pip install -r requirements.txt
	python -m pip install datashader==0.15.2
	python -m pip install "holoviews[recommended]"
	python -m pip install -e .

setup:
	python -m pip install -e .

test:
	pytest

fmt:
	python -m black happy projects analysis qupath