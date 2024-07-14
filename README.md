### Setup
1. **Set up a virtual environment (tested with Python 3.8-3.11)**  

    ```sh
    conda create --name TOKEN python=3.8
    ```

2. **Install required dependencies**  

    ```sh
    pip install -r requirements.txt.lock
    ```
### Launch training job with sample reasoning data

    ```sh
    python -m torch.distributed.launch --nproc_per_node=2 train.py
    ```