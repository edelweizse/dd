# Data links:<br>
CTD:<br>
[chem-gene](https://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz)<br>
[chem-ds](https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz)<br>
[chem-ds-currated](https://ctdbase.org/reports/CTD_curated_chemicals_diseases.tsv.gz)<br>
[gene-ds](https://ctdbase.org/reports/CTD_curated_genes_diseases.tsv.gz)<br>
SNAP:<br>
[ppi](https://snap.stanford.edu/biodata/datasets/10008/files/PP-Decagon_ppi.csv.gz)<br>

# Usage

### Process raw data:
```bash
python -m scripts.process_data --raw-dir ./data/raw --processed-dir ./data/processed
```

### Train model:
```bash
python -m scripts.train --processed-dir ./data/processed --hidden-dim 256 --num-layers 3 --num-heads 4 --dropout 0.003 --epochs 200 --batch-size 4096 --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --num-neg-train 20 --num-neg-eval 20 --val-ratio 0.1 --test-ratio 0.1 --patience 10 --factor 0.5 --early-stopping 20 --ckpt-dir ./checkpoints --run-name hgt_cd_lp --experiment-name HGT_linkpred
```

python -m scripts.train --processed-dir ./data/processed --hidden-dim 128 --num-layers 2 --num-heads 4 --dropout 0.002 --epochs 200 --batch-size 2048 --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --num-neg-train 10 --num-neg-eval 10 --val-ratio 0.1 --test-ratio 0.1 --patience 10 --factor 0.5 --early-stopping 20 --ckpt-dir ./checkpoints --run-name hgt_cd_lp --experiment-name HGT_linkpred

### Predictions:
```bash
python -m scripts.predict --disease MESH:D003920 --chemical D008687
python -m scripts.predict --disease MESH:D003920 --top-k 10
python -m scripts.predict --chemical D008687 --top-k 10
```