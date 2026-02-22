"""
Training module for chemical-disease link prediction.

This module provides:
- Checkpoint saving/loading utilities
- Main training loop with MLflow logging
- Early stopping and learning rate scheduling
- Optuna-compatible training function for hyperparameter tuning
"""

import os
import time
import math
import torch
import torch.nn.functional as F
import mlflow
from typing import Tuple, Optional, Dict, Any

from src.data.splits import SplitArtifacts, negative_sample_cd_batch_local
from src.models.architectures.hgat import HGATPredictor
from .utils import bce_with_logits, eval_epoch

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _seeded_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a device-compatible generator for deterministic sampling."""
    if device.type == 'cuda':
        g = torch.Generator(device=device)
    else:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def save_checkpoint(
    path: str,
    model: HGATPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metrics: float
):
    """
    Save model checkpoint.
    
    Args:
        path: Path to save checkpoint.
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        epoch: Current epoch.
        best_metrics: Best validation metric value.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict(),
        'best_metrics': best_metrics
    }, path)


def load_checkpoint(
    path: str,
    model: HGATPredictor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: torch.device = None
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to map checkpoint to.
        
    Returns:
        Checkpoint dictionary with metadata.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    if scheduler is not None and 'sched_state' in ckpt:
        scheduler.load_state_dict(ckpt['sched_state'])
    return ckpt


def train(
    arts: SplitArtifacts,
    model: HGATPredictor,
    device: torch.device,
    *,
    run_name: str = 'hgat_cd_linkpred',
    experiment_name: str = 'CTD_HGAT',
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    num_neg_train: int = 5,
    num_neg_eval: int = 20,
    ks: Tuple[int, ...] = (5, 10, 50),
    amp: bool = True,
    ckpt_dir: str = './checkpoints',
    monitor: str = 'auprc',
    patience: int = 5,
    factor: float = 0.5,
    early_stopping_patience: int = 10,
    pos_weight: Optional[float] = None,
    focal_gamma: float = 0.0,
    hard_negative_ratio: float = 0.5,
    degree_alpha: float = 0.75,
    eval_hard_negative_ratio: float = 0.0
) -> HGATPredictor:
    """
    Train the model with MLflow logging.
    
    Args:
        arts: SplitArtifacts containing data loaders and split info.
        model: HGATPredictor model to train.
        device: Torch device.
        run_name: MLflow run name.
        experiment_name: MLflow experiment name.
        epochs: Maximum number of epochs.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient clipping value.
        num_neg_train: Number of negatives per positive during training.
        num_neg_eval: Number of negatives per positive during evaluation.
        ks: Tuple of K values for Hits@K metrics.
        amp: Whether to use automatic mixed precision.
        ckpt_dir: Directory for checkpoints.
        monitor: Metric to monitor for model selection.
        patience: LR scheduler patience.
        factor: LR scheduler reduction factor.
        early_stopping_patience: Epochs without improvement before stopping.
        pos_weight: Optional positive-class reweighting factor for BCE.
            If None, defaults to num_neg_train.
        focal_gamma: Focal loss gamma. Set > 0 to enable focal BCE.
        hard_negative_ratio: Fraction of degree-biased negatives in training.
        degree_alpha: Exponent for degree-biased negative sampling.
        eval_hard_negative_ratio: Fraction of degree-biased negatives for
            sampled evaluation.
        
    Returns:
        Trained model (loaded from best checkpoint).
    """
    valid_metrics = ['auroc', 'auprc', 'mrr'] + [f'hits_{k}' for k in ks]
    assert monitor in valid_metrics, f'Monitor must be one of {valid_metrics}, got {monitor}.'
    
    model = model.to(device)
    effective_pos_weight = float(num_neg_train) if pos_weight is None else float(pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience
    )
    
    scaler = torch.amp.GradScaler(enabled=(amp and device.type == 'cuda'))
    split_seed = int((arts.split_metadata or {}).get('seed', 42))
    train_neg_generator = _seeded_generator(split_seed + 101, device)
    train_known_pos = getattr(arts, 'known_pos', arts.known_pos_test)
    val_sampling_seed = split_seed + 202
    test_sampling_seed = split_seed + 303
    
    best_val = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'model': 'HGATPredictor',
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'grad_clip': grad_clip,
            'num_neg_train': num_neg_train,
            'num_neg_eval': num_neg_eval,
            'amp': amp,
            'monitor': monitor,
            'patience': patience,
            'factor': factor,
            'pos_weight': effective_pos_weight,
            'focal_gamma': focal_gamma,
            'hard_negative_ratio': hard_negative_ratio,
            'degree_alpha': degree_alpha,
            'eval_hard_negative_ratio': eval_hard_negative_ratio,
            'neighbours': getattr(arts.train_loader, 'num_neighbors', None),
            'batch_size': getattr(arts.train_loader, 'batch_size', None)
        })
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            
            loss_sum = 0.0
            n_pos_sum = 0
            
            for batch in arts.train_loader:
                batch = batch.to(device)
                
                # Access edge_label_index from the batch
                cd_edge_store = batch[('chemical', 'associated_with', 'disease')]
                pos_edge = cd_edge_store.edge_label_index  # [2, B] local
                
                neg_edge = negative_sample_cd_batch_local(
                    batch_data=batch,
                    pos_edge_index_local=pos_edge,
                    known_pos=train_known_pos,
                    num_neg_per_pos=num_neg_train,
                    hard_negative_ratio=hard_negative_ratio,
                    degree_alpha=degree_alpha,
                    global_chem_degree=arts.chem_train_degree,
                    global_dis_degree=arts.dis_train_degree,
                    generator=train_neg_generator,
                )
                
                optimizer.zero_grad()
                if amp and device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pos_logits, neg_logits = model(batch, pos_edge, neg_edge)
                        loss = bce_with_logits(
                            pos_logits,
                            neg_logits,
                            pos_weight=effective_pos_weight,
                            focal_gamma=focal_gamma
                        )
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pos_logits, neg_logits = model(batch, pos_edge, neg_edge)
                    loss = bce_with_logits(
                        pos_logits,
                        neg_logits,
                        pos_weight=effective_pos_weight,
                        focal_gamma=focal_gamma
                    )
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                
                B = pos_edge.size(1)
                loss_sum += float(loss.item()) * B
                n_pos_sum += B
            
            train_loss = loss_sum / max(n_pos_sum, 1)
            
            val_metrics = eval_epoch(
                model,
                arts.val_loader,
                arts.known_pos_val,
                device,
                num_neg_eval,
                ks,
                amp,
                hard_negative_ratio=eval_hard_negative_ratio,
                degree_alpha=degree_alpha,
                sampling_seed=val_sampling_seed,
                global_chem_degree=arts.chem_train_degree,
                global_dis_degree=arts.dis_train_degree,
            )
            
            scheduler.step(val_metrics[monitor])
            curr_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - t0
            log = {
                'train_loss': train_loss,
                'lr': curr_lr,
                'epoch_time_sec': epoch_time
            }
            log.update({f'val_{k}': float(v) for k, v in val_metrics.items()})
            mlflow.log_metrics(log, step=epoch)
            
            last_path = os.path.join(ckpt_dir, 'last.pt')
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val)
            
            score = float(val_metrics[monitor])
            if score > best_val:
                best_val = score
                best_epoch = epoch
                best_path = os.path.join(ckpt_dir, 'best.pt')
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val)
                mlflow.log_artifact(best_path)
                print(f'New best {monitor}: {best_val:.4f} at epoch {epoch}.')
                epochs_without_improvement = 0
            else:
                print(f'No improvement in {monitor} for epoch {epoch}.')
                epochs_without_improvement += 1
            
            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
            
            print(
                f'EP = {epoch:03d}  train_loss = {train_loss:.4f} '
                f'val_auroc = {val_metrics["auroc"]:.4f}  val_auprc = {val_metrics["auprc"]:.4f} '
                f'val_mrr = {val_metrics["mrr"]:.4f} val_hits_10 = {val_metrics.get("hits_10", float("nan")):.4f} '
                f'lr = {curr_lr:.2e}  time = {epoch_time:.1f}s'
            )
        
        # Load best model and evaluate on test set
        best_path = os.path.join(ckpt_dir, 'best.pt')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
        
        test_metrics = eval_epoch(
            model,
            arts.test_loader,
            arts.known_pos_test,
            device,
            num_neg_eval,
            ks,
            amp,
            hard_negative_ratio=eval_hard_negative_ratio,
            degree_alpha=degree_alpha,
            sampling_seed=test_sampling_seed,
            global_chem_degree=arts.chem_train_degree,
            global_dis_degree=arts.dis_train_degree,
        )
        mlflow.log_metrics({f'test_{k}': float(v) for k, v in test_metrics.items()})
        
        print(f'Best ep = {best_epoch} val_{monitor} = {best_val:.4f}')
        print("TEST:\n", {k: round(float(v), 6) for k, v in test_metrics.items()})
    
    return model


def train_for_tuning(
    trial: 'optuna.Trial',
    arts: SplitArtifacts,
    model: HGATPredictor,
    device: torch.device,
    *,
    run_name: str,
    experiment_name: str = 'HGAT_tuning',
    epochs: int = 25,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    num_neg_train: int = 5,
    num_neg_eval: int = 20,
    ks: Tuple[int, ...] = (10,),
    amp: bool = True,
    ckpt_dir: Optional[str] = None,
    monitor: str = 'auprc',
    patience: int = 5,
    factor: float = 0.5,
    early_stopping_patience: int = 7,
    hyperparams: Optional[Dict[str, Any]] = None,
    pos_weight: Optional[float] = None,
    focal_gamma: float = 0.0,
    hard_negative_ratio: float = 0.5,
    degree_alpha: float = 0.75,
    eval_hard_negative_ratio: float = 0.0
) -> float:
    """
    Train the model with Optuna pruning support and MLflow logging.
    
    This function is designed for hyperparameter tuning:
    - Reports validation metrics to Optuna for pruning decisions
    - Uses shorter training runs with aggressive early stopping
    - Logs all trials to MLflow for comparison
    - Handles OOM errors gracefully
    
    Args:
        trial: Optuna Trial object for pruning decisions.
        arts: SplitArtifacts containing data loaders and split info.
        model: HGATPredictor model to train.
        device: Torch device.
        run_name: MLflow run name (usually includes trial number).
        experiment_name: MLflow experiment name.
        epochs: Maximum number of epochs (shorter for tuning).
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient clipping value.
        num_neg_train: Number of negatives per positive during training.
        num_neg_eval: Number of negatives per positive during evaluation.
        ks: Tuple of K values for Hits@K metrics.
        amp: Whether to use automatic mixed precision.
        ckpt_dir: Optional checkpoint directory for this trial.
            When set, writes ``last.pt`` every epoch and ``best.pt`` on improvement.
        monitor: Metric to monitor for model selection.
        patience: LR scheduler patience.
        factor: LR scheduler reduction factor.
        early_stopping_patience: Epochs without improvement before stopping.
        hyperparams: Optional dict of hyperparameters to log.
        pos_weight: Optional positive-class reweighting factor for BCE.
            If None, defaults to num_neg_train.
        focal_gamma: Focal loss gamma. Set > 0 to enable focal BCE.
        hard_negative_ratio: Fraction of degree-biased negatives in training.
        degree_alpha: Exponent for degree-biased negative sampling.
        eval_hard_negative_ratio: Fraction of degree-biased negatives for
            sampled evaluation.
        
    Returns:
        Best validation metric (AUPRC by default).
        
    Raises:
        optuna.TrialPruned: If the trial is pruned by Optuna.
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is required for train_for_tuning. Install with: pip install optuna")
    
    model = model.to(device)
    effective_pos_weight = float(num_neg_train) if pos_weight is None else float(pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience
    )
    
    scaler = torch.amp.GradScaler(enabled=(amp and device.type == 'cuda'))
    split_seed = int((arts.split_metadata or {}).get('seed', 42))
    train_neg_generator = _seeded_generator(split_seed + 404, device)
    train_known_pos = getattr(arts, 'known_pos', arts.known_pos_test)
    val_sampling_seed = split_seed + 505
    
    best_val = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    last_path = os.path.join(ckpt_dir, 'last.pt') if ckpt_dir else None
    best_path = os.path.join(ckpt_dir, 'best.pt') if ckpt_dir else None
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        params_to_log = {
            'model': 'HGATPredictor',
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'grad_clip': grad_clip,
            'num_neg_train': num_neg_train,
            'num_neg_eval': num_neg_eval,
            'amp': amp,
            'monitor': monitor,
            'patience': patience,
            'factor': factor,
            'early_stopping_patience': early_stopping_patience,
            'pos_weight': effective_pos_weight,
            'focal_gamma': focal_gamma,
            'hard_negative_ratio': hard_negative_ratio,
            'degree_alpha': degree_alpha,
            'eval_hard_negative_ratio': eval_hard_negative_ratio,
            'trial_number': trial.number,
            'batch_size': getattr(arts.train_loader, 'batch_size', None),
            'ckpt_dir': ckpt_dir,
        }
        
        # Add sampled hyperparameters if provided
        if hyperparams:
            params_to_log.update(hyperparams)
        
        mlflow.log_params(params_to_log)
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            
            loss_sum = 0.0
            n_pos_sum = 0
            
            for batch in arts.train_loader:
                batch = batch.to(device)
                
                cd_edge_store = batch[('chemical', 'associated_with', 'disease')]
                pos_edge = cd_edge_store.edge_label_index
                
                neg_edge = negative_sample_cd_batch_local(
                    batch_data=batch,
                    pos_edge_index_local=pos_edge,
                    known_pos=train_known_pos,
                    num_neg_per_pos=num_neg_train,
                    hard_negative_ratio=hard_negative_ratio,
                    degree_alpha=degree_alpha,
                    global_chem_degree=arts.chem_train_degree,
                    global_dis_degree=arts.dis_train_degree,
                    generator=train_neg_generator,
                )
                
                optimizer.zero_grad()
                if amp and device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pos_logits, neg_logits = model(batch, pos_edge, neg_edge)
                        loss = bce_with_logits(
                            pos_logits,
                            neg_logits,
                            pos_weight=effective_pos_weight,
                            focal_gamma=focal_gamma
                        )
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pos_logits, neg_logits = model(batch, pos_edge, neg_edge)
                    loss = bce_with_logits(
                        pos_logits,
                        neg_logits,
                        pos_weight=effective_pos_weight,
                        focal_gamma=focal_gamma
                    )
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                
                B = pos_edge.size(1)
                loss_sum += float(loss.item()) * B
                n_pos_sum += B
            
            train_loss = loss_sum / max(n_pos_sum, 1)
            
            # Validation
            val_metrics = eval_epoch(
                model,
                arts.val_loader,
                arts.known_pos_val,
                device,
                num_neg_eval,
                ks,
                amp,
                hard_negative_ratio=eval_hard_negative_ratio,
                degree_alpha=degree_alpha,
                sampling_seed=val_sampling_seed,
                global_chem_degree=arts.chem_train_degree,
                global_dis_degree=arts.dis_train_degree,
            )
            
            scheduler.step(val_metrics[monitor])
            curr_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - t0
            
            # Log to MLflow
            log = {
                'train_loss': train_loss,
                'lr': curr_lr,
                'epoch_time_sec': epoch_time
            }
            log.update({f'val_{k}': float(v) for k, v in val_metrics.items()})
            mlflow.log_metrics(log, step=epoch)

            if last_path is not None:
                save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val)
            
            # Track best score
            score = float(val_metrics[monitor])
            if score > best_val:
                best_val = score
                best_epoch = epoch
                epochs_without_improvement = 0
                if best_path is not None:
                    save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val)
                    mlflow.log_artifact(best_path)
            else:
                epochs_without_improvement += 1
            
            # Report to Optuna for pruning
            trial.report(score, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                mlflow.log_metric('pruned', 1)
                mlflow.log_metric('pruned_at_epoch', epoch)
                raise optuna.TrialPruned()
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                break
            
            print(
                f'[Trial {trial.number}] EP {epoch:02d}  '
                f'loss={train_loss:.4f}  val_{monitor}={score:.4f}  '
                f'best={best_val:.4f}  lr={curr_lr:.2e}  time={epoch_time:.1f}s'
            )
        
        # Log final metrics
        mlflow.log_metric(f'best_val_{monitor}', best_val)
        mlflow.log_metric('best_epoch', best_epoch)
        mlflow.log_metric('total_epochs', epoch)
        mlflow.log_metric('pruned', 0)
    
    return best_val
