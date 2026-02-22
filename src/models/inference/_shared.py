"""Shared helpers for inference predictors."""

from __future__ import annotations

from typing import Any, Callable, Dict

import polars as pl
import torch


def build_id_mappings(
    disease_df: pl.DataFrame,
    chemical_df: pl.DataFrame,
) -> Dict[str, Any]:
    """Build internal/external ID and display-name mappings."""
    disease_to_id = dict(
        zip(
            disease_df['DS_OMIM_MESH_ID'].to_list(),
            disease_df['DS_ID'].to_list(),
        )
    )
    id_to_disease = dict(
        zip(
            disease_df['DS_ID'].to_list(),
            disease_df['DS_OMIM_MESH_ID'].to_list(),
        )
    )
    disease_names = dict(
        zip(
            disease_df['DS_OMIM_MESH_ID'].to_list(),
            disease_df['DS_NAME'].to_list(),
        )
    )

    chemical_to_id = dict(
        zip(
            chemical_df['CHEM_MESH_ID'].to_list(),
            chemical_df['CHEM_ID'].to_list(),
        )
    )
    id_to_chemical = dict(
        zip(
            chemical_df['CHEM_ID'].to_list(),
            chemical_df['CHEM_MESH_ID'].to_list(),
        )
    )
    chemical_names = dict(
        zip(
            chemical_df['CHEM_MESH_ID'].to_list(),
            chemical_df['CHEM_NAME'].to_list(),
        )
    )

    return {
        'disease_to_id': disease_to_id,
        'id_to_disease': id_to_disease,
        'disease_names': disease_names,
        'chemical_to_id': chemical_to_id,
        'id_to_chemical': id_to_chemical,
        'chemical_names': chemical_names,
        'num_diseases': len(disease_to_id),
        'num_chemicals': len(chemical_to_id),
    }


def bilinear_score(
    z_chem: torch.Tensor,
    z_dis: torch.Tensor,
    decoder_weight: torch.Tensor,
    chem_ids: torch.Tensor,
    dis_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute bilinear logits for (chemical, disease) pairs."""
    c = z_chem[chem_ids]
    d = z_dis[dis_ids]
    return (c @ decoder_weight * d).sum(dim=-1)


def predict_pair_common(
    *,
    disease_id: str,
    chemical_id: str,
    disease_to_id: Dict[str, int],
    chemical_to_id: Dict[str, int],
    disease_names: Dict[str, str],
    chemical_names: Dict[str, str],
    device: torch.device,
    threshold: float,
    compute_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    is_known_link: Callable[[int, int], bool],
) -> Dict[str, Any]:
    """Shared pair-prediction flow with validation and output formatting."""
    if disease_id not in disease_to_id:
        raise ValueError(f"Unknown disease ID: {disease_id}")
    if chemical_id not in chemical_to_id:
        raise ValueError(f"Unknown chemical ID: {chemical_id}")

    dis_idx = disease_to_id[disease_id]
    chem_idx = chemical_to_id[chemical_id]

    chem_tensor = torch.tensor([chem_idx], device=device)
    dis_tensor = torch.tensor([dis_idx], device=device)

    with torch.no_grad():
        logit = compute_score(chem_tensor, dis_tensor).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    return {
        'disease_id': disease_id,
        'chemical_id': chemical_id,
        'disease_name': disease_names.get(disease_id, 'Unknown'),
        'chemical_name': chemical_names.get(chemical_id, 'Unknown'),
        'probability': prob,
        'label': int(prob >= threshold),
        'logit': logit,
        'known': is_known_link(chem_idx, dis_idx),
    }


def rank_chemicals_for_disease_common(
    *,
    disease_id: str,
    disease_to_id: Dict[str, int],
    id_to_chemical: Dict[int, str],
    chemical_names: Dict[str, str],
    num_chemicals: int,
    device: torch.device,
    top_k: int,
    exclude_known: bool,
    compute_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    is_known_link: Callable[[int, int], bool],
) -> pl.DataFrame:
    """Shared top-k chemical ranking for a disease."""
    if disease_id not in disease_to_id:
        raise ValueError(f"Unknown disease ID: {disease_id}")

    dis_idx = disease_to_id[disease_id]
    all_chem_ids = torch.arange(num_chemicals, device=device)
    dis_ids = torch.full((num_chemicals,), dis_idx, device=device)

    with torch.no_grad():
        logits = compute_score(all_chem_ids, dis_ids)
        probs = torch.sigmoid(logits)

    sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()
    results = []
    for idx in sorted_indices:
        is_known = is_known_link(idx, dis_idx)
        if exclude_known and is_known:
            continue

        chem_mesh_id = id_to_chemical[idx]
        results.append({
            'rank': len(results) + 1,
            'chemical_id': chem_mesh_id,
            'chemical_name': chemical_names.get(chem_mesh_id, 'Unknown'),
            'probability': probs[idx].item(),
            'logit': logits[idx].item(),
            'known': is_known,
        })
        if len(results) >= top_k:
            break

    return pl.DataFrame(results)


def rank_diseases_for_chemical_common(
    *,
    chemical_id: str,
    chemical_to_id: Dict[str, int],
    id_to_disease: Dict[int, str],
    disease_names: Dict[str, str],
    num_diseases: int,
    device: torch.device,
    top_k: int,
    exclude_known: bool,
    compute_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    is_known_link: Callable[[int, int], bool],
) -> pl.DataFrame:
    """Shared top-k disease ranking for a chemical."""
    if chemical_id not in chemical_to_id:
        raise ValueError(f"Unknown chemical ID: {chemical_id}")

    chem_idx = chemical_to_id[chemical_id]
    all_dis_ids = torch.arange(num_diseases, device=device)
    chem_ids = torch.full((num_diseases,), chem_idx, device=device)

    with torch.no_grad():
        logits = compute_score(chem_ids, all_dis_ids)
        probs = torch.sigmoid(logits)

    sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()
    results = []
    for idx in sorted_indices:
        is_known = is_known_link(chem_idx, idx)
        if exclude_known and is_known:
            continue

        dis_mesh_id = id_to_disease[idx]
        results.append({
            'rank': len(results) + 1,
            'disease_id': dis_mesh_id,
            'disease_name': disease_names.get(dis_mesh_id, 'Unknown'),
            'probability': probs[idx].item(),
            'logit': logits[idx].item(),
            'known': is_known,
        })
        if len(results) >= top_k:
            break

    return pl.DataFrame(results)
