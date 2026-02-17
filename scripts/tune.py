#!/usr/bin/env python
"""
Hyperparameter tuning with Optuna for HGT chemical-disease link prediction.

This script performs Bayesian hyperparameter optimization using Optuna with:
- TPE (Tree-structured Parzen Estimator) sampler for smart parameter search
- MedianPruner to stop unpromising trials early
- MLflow logging for all trials
- SQLite storage for resumability
- OOM error handling

Usage:
    # Run for 4 hours (default)
    python scripts/tune.py --timeout 14400
    
    # Run for specific number of trials
    python scripts/tune.py --n-trials 30
    
    # Resume a previous study
    python scripts/tune.py --study-name my_study --timeout 7200
    
    # Custom search with smaller model space (for testing)
    python scripts/tune.py --timeout 3600 --quick
"""

import argparse
import json
import gc
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.cli_config import parse_args_with_config
from src.data.graph import build_graph_from_processed, print_graph_summary
from src.data.splits import prepare_splits_and_loaders
from src.models.hgt import HGTPredictor
from src.training.trainer import train_for_tuning


# ============================================================================
# SEARCH SPACE CONFIGURATION
# ============================================================================

SEARCH_SPACE_FULL = {
    'hidden_dim': [128, 192, 256, 384],
    'num_layers': (2, 4),
    'num_heads': [4, 6, 8],
    'dropout': (0.05, 0.25),
    'batch_size': [1024, 2048, 3072],
    'lr': (5e-5, 5e-4),
    'weight_decay': (1e-5, 1e-3),
    'num_neg_train': [5, 10, 15],
    'num_neighbours_options': [[5, 3], [8, 4], [10, 5]],
}

SEARCH_SPACE_QUICK = {
    'hidden_dim': [128, 192, 256],
    'num_layers': (2, 3),
    'num_heads': [4, 6],
    'dropout': (0.08, 0.20),
    'batch_size': [1024, 2048],
    'lr': (1e-4, 4e-4),
    'weight_decay': (5e-5, 5e-4),
    'num_neg_train': [5, 10],
    'num_neighbours_options': [[8, 4], [10, 5]],
}


def sample_hyperparams(trial: optuna.Trial, search_space: Dict) -> Dict[str, Any]:
    """Sample hyperparameters from the search space."""
    
    hidden_dim = trial.suggest_categorical('hidden_dim', search_space['hidden_dim'])
    num_layers = trial.suggest_int('num_layers', *search_space['num_layers'])
    num_heads = trial.suggest_categorical('num_heads', search_space['num_heads'])
    dropout = trial.suggest_float('dropout', *search_space['dropout'])

    batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])
    lr = trial.suggest_float('lr', *search_space['lr'], log=True)
    weight_decay = trial.suggest_float('weight_decay', *search_space['weight_decay'], log=True)
    num_neg_train = trial.suggest_categorical('num_neg_train', search_space['num_neg_train'])
    
    # Neighbor sampling
    num_neighbours_idx = trial.suggest_int(
        'num_neighbours_idx', 
        0, 
        len(search_space['num_neighbours_options']) - 1
    )
    num_neighbours = search_space['num_neighbours_options'][num_neighbours_idx]
    
    return {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'num_neg_train': num_neg_train,
        'num_neighbours': num_neighbours,
        'num_neighbours_idx': num_neighbours_idx,
    }


def create_objective(
    data,
    vocabs,
    device,
    search_space: Dict,
    experiment_name: str,
    epochs_per_trial: int = 25,
    early_stopping_patience: int = 7,
):
    """
    Create an Optuna objective function with access to shared data.
    
    Using closure to avoid reloading data for each trial.
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Single trial objective function."""

        params = sample_hyperparams(trial, search_space)
        
        run_name = f"trial_{trial.number:03d}_{datetime.now().strftime('%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"{'='*60}")
        print(f"Parameters: {json.dumps({k: v for k, v in params.items() if k != 'num_neighbours_idx'}, indent=2)}")
        
        try:
            arts = prepare_splits_and_loaders(
                data_full=data,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
                batch_size=params['batch_size'],
                num_neighbours=params['num_neighbours']
            )
            
            num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
            node_input_dims = {
                ntype: int(data[ntype].x.size(1))
                for ntype in data.node_types
                if isinstance(data[ntype].x, torch.Tensor)
                and data[ntype].x.dim() == 2
                and data[ntype].x.is_floating_point()
            }
            
            model = HGTPredictor(
                num_nodes_dict=num_nodes_dict,
                metadata=arts.data_train.metadata(),
                node_input_dims=node_input_dims,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                num_heads=params['num_heads'],
                dropout=params['dropout'],
                num_action_types=vocabs['action_type'].height,
                num_action_subjects=vocabs['action_subject'].height
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
            
            best_auprc = train_for_tuning(
                trial=trial,
                arts=arts,
                model=model,
                device=device,
                run_name=run_name,
                experiment_name=experiment_name,
                epochs=epochs_per_trial,
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                grad_clip=1.0,
                num_neg_train=params['num_neg_train'],
                num_neg_eval=20,
                amp=True,
                monitor='auprc',
                patience=5,
                factor=0.5,
                early_stopping_patience=early_stopping_patience,
                hyperparams=params
            )
            
            print(f"Trial {trial.number} finished with val_auprc={best_auprc:.4f}")
            
            return best_auprc
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"Trial {trial.number} OOM - pruning")
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()
            raise
        
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    return objective


def print_study_summary(study: optuna.Study):
    """Print a summary of the optimization study."""
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal trials: {len(study.trials)}")
    
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    print(f"  Completed: {completed}")
    print(f"  Pruned: {pruned}")
    print(f"  Failed: {failed}")
    
    if completed > 0:
        print(f"\nBest trial:")
        print(f"  Number: {study.best_trial.number}")
        print(f"  Value (val_auprc): {study.best_value:.4f}")
        print(f"\nBest parameters:")
        for key, value in study.best_params.items():
            if key == 'num_neighbours_idx':
                continue
            print(f"  {key}: {value}")
        
        # Show num_neighbours from the index
        if 'num_neighbours_idx' in study.best_params:
            idx = study.best_params['num_neighbours_idx']
            options = SEARCH_SPACE_FULL['num_neighbours_options']
            if idx < len(options):
                print(f"  num_neighbours: {options[idx]}")
    else:
        print("\nNo completed trials yet.")
    
    print("="*60)


def save_results(study: optuna.Study, output_dir: Path):
    """Save study results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
        best_params = dict(study.best_params)
        
        # Convert num_neighbours_idx to actual values
        if 'num_neighbours_idx' in best_params:
            idx = best_params.pop('num_neighbours_idx')
            options = SEARCH_SPACE_FULL['num_neighbours_options']
            if idx < len(options):
                best_params['num_neighbours'] = options[idx]
        
        results = {
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'best_params': best_params,
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
        
        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir / 'best_params.json'}")
        
        cmd = generate_train_command(best_params)
        with open(output_dir / 'train_command.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Best parameters from Optuna tuning\n")
            f.write(f"# Best val_auprc: {study.best_value:.4f}\n\n")
            f.write(cmd)
        
        print(f"Training command saved to {output_dir / 'train_command.sh'}")


def generate_train_command(params: Dict) -> str:
    """Generate a train.py command from best parameters."""
    
    cmd_parts = ["python scripts/train.py"]
    
    param_mapping = {
        'hidden_dim': '--hidden-dim',
        'num_layers': '--num-layers',
        'num_heads': '--num-heads',
        'dropout': '--dropout',
        'batch_size': '--batch-size',
        'lr': '--lr',
        'weight_decay': '--weight-decay',
        'num_neg_train': '--num-neg-train',
    }
    
    for param, flag in param_mapping.items():
        if param in params:
            value = params[param]
            if isinstance(value, float):
                cmd_parts.append(f"    {flag} {value:.6g}")
            else:
                cmd_parts.append(f"    {flag} {value}")
    
    # Handle num_neighbours specially
    if 'num_neighbours' in params:
        neighbours = params['num_neighbours']
        cmd_parts.append(f"    --num-neighbours {neighbours[0]} {neighbours[1]}")
    
    cmd_parts.append("    --epochs 100")
    cmd_parts.append("    --early-stopping 15")
    cmd_parts.append("    --patience 7")
    
    return " \\\n".join(cmd_parts) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for HGT chemical-disease link prediction'
    )
    
    # Optimization settings
    parser.add_argument('--timeout', type=int, default=14400,
                        help='Time limit in seconds (default: 4 hours = 14400)')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Maximum number of trials (optional, use with or instead of timeout)')
    
    # Study settings
    parser.add_argument('--study-name', type=str, default='hgt_cd_tuning',
                        help='Optuna study name (for resuming)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_hgt.db',
                        help='Optuna storage URL')
    parser.add_argument('--experiment-name', type=str, default='HGT_tuning',
                        help='MLflow experiment name')
    
    # Data settings
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--use-node-features', action='store_true',
                        help='Use precomputed node feature tables for inductive tuning')
    parser.add_argument('--node-features-dir', type=str, default=None,
                        help='Directory with node feature parquet files')
    
    # Training settings per trial
    parser.add_argument('--epochs-per-trial', type=int, default=25,
                        help='Maximum epochs per trial')
    parser.add_argument('--early-stopping', type=int, default=7,
                        help='Early stopping patience per trial')
    
    # Search space settings
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller search space for faster iteration')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='./tuning_results',
                        help='Directory to save results')
    
    args, _ = parse_args_with_config(parser)
    
    # Select search space
    search_space = SEARCH_SPACE_QUICK if args.quick else SEARCH_SPACE_FULL
    space_name = "QUICK" if args.quick else "FULL"
    
    print(f"\n{'='*60}")
    print("HGT Hyperparameter Tuning with Optuna")
    print(f"{'='*60}")
    print(f"Search space: {space_name}")
    print(f"Timeout: {args.timeout}s ({args.timeout/3600:.1f} hours)")
    if args.n_trials:
        print(f"Max trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"Early stopping: {args.early_stopping}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"MLflow experiment: {args.experiment_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data once (shared across all trials)
    print(f"\nLoading graph data from {args.processed_dir}...")
    data, vocabs = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=True,
        include_extended=True,
        use_node_features=args.use_node_features,
        node_features_dir=args.node_features_dir
    )
    print_graph_summary(data)
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Check if resuming
    if len(study.trials) > 0:
        print(f"\nResuming study with {len(study.trials)} existing trials")
        print(f"Current best: {study.best_value:.4f}")
    
    objective = create_objective(
        data=data,
        vocabs=vocabs,
        device=device,
        search_space=search_space,
        experiment_name=args.experiment_name,
        epochs_per_trial=args.epochs_per_trial,
        early_stopping_patience=args.early_stopping
    )

    print(f"\nStarting optimization...")
    print(f"{'='*60}\n")
    
    try:
        study.optimize(
            objective,
            timeout=args.timeout,
            n_trials=args.n_trials,
            show_progress_bar=True,
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
    
    print_study_summary(study)
    save_results(study, Path(args.output_dir))
    
    print(f"\nTo visualize results, you can use:")
    print(f"  optuna-dashboard {args.storage}")
    print(f"\nOr in Python:")
    print(f"  import optuna")
    print(f"  study = optuna.load_study('{args.study_name}', '{args.storage}')")
    print(f"  optuna.visualization.plot_optimization_history(study)")


if __name__ == '__main__':
    main()
