## How to run the pipeline

Quick test run
```powershell
cd "C:\Users\under\Desktop\sparsify\data"
python cp_pipeline.py --data_dir . --rank 7 --n_inits 1 --n_outer 10 --n_iter 200
```



### Arguments
| Argument | What it does | Default |
|---|---|---|
| --data_dir | Folder with .npz files | . |
| --rank | Number of CP components | 10 |
| --n_inits | Random restarts | 3 |
| --n_outer | Imputation loops | 10 |
| --n_iter | Iterations per init | 200 |
