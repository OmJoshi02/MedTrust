"""
MedTrust V1 — Cosine Similarity and Case-Ranking Calculator (src/similarity/similarity_calculator.py)

Module 8: Similarity Analysis.
Implements L2-normalized cosine similarity computation and Top-K similar case ranking.

Mathematical Formulation:
    Cosine Similarity = (q · r) / (||q||_2 * ||r||_2)
When vectors are L2-normalized unit vectors (||q|| = ||r|| = 1), this reduces to:
    Cosine Similarity = q · r

Scores range theoretically between -1.0 and 1.0. For positive feature representations,
we scale to percentage [0.00%, 100.00%] rounded to 2 decimal places.

SCIENTIFIC INTEGRITY & MEDICAL DISCLAIMER:
Similarity scores represent mathematical image feature proximity only.
They are NOT disease probabilities, confidence scores, or diagnostic metrics.
A high similarity score does NOT guarantee that two radiographs share the same pathology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union
import numpy as np

from src.similarity.disease_frequency import CANONICAL_DISEASES, NO_FINDING_LABEL

# Non-diagnostic medical disclaimer mandated by AGENTS.md Rule 18
DISCLAIMER_TEXT: str = (
    "Similarity scores represent mathematical image/feature similarity only. "
    "They are not a medical diagnosis or clinical recommendation. "
    "Case similarity provides supporting context, not proof of identical pathology."
)

EXPLANATION_TEXT: str = (
    "NIH Dataset Prevalence measures label frequency across the dataset. "
    "Retrieved-Case Label Support measures label representation among visually similar references. "
    "Neither is a calibrated disease probability."
)


def compute_cosine_similarity(
    query_vector: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and N reference vectors.

    Args:
        query_vector: 1D array of shape (D,) or 2D array of shape (1, D).
        reference_embeddings: 2D array of shape (N, D) containing reference vectors.

    Returns:
        1D array of length N with similarity scores scaled to [0.00, 100.00].
    """
    if query_vector.ndim == 1:
        q = query_vector.reshape(1, -1)
    else:
        q = query_vector

    if reference_embeddings.ndim != 2:
        raise ValueError(
            f"reference_embeddings must be a 2D array (N, D), got shape {reference_embeddings.shape}"
        )

    if q.shape[1] != reference_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {q.shape[1]} features, "
            f"references have {reference_embeddings.shape[1]} features."
        )

    # Compute L2 norms with division-by-zero protection
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)

    q_norm = np.where(q_norm == 0, 1e-12, q_norm)
    ref_norm = np.where(ref_norm == 0, 1e-12, ref_norm)

    normalized_q = q / q_norm
    normalized_ref = reference_embeddings / ref_norm

    # Dot product calculation (N, 1) -> (N,)
    cosine_sim = np.dot(normalized_ref, normalized_q.T).flatten()

    # Clip to valid cosine range [-1.0, 1.0] to prevent floating point overshoot
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    # Convert to non-negative percentage [0.0, 100.0]
    percentage_scores = np.maximum(0.0, cosine_sim) * 100.0

    return percentage_scores.astype(np.float64)


def normalize_disease_name(name: str) -> str:
    """Normalize disease name for robust matching (e.g., 'Pleural_Thickening' <-> 'Pleural Thickening')."""
    return str(name).strip().lower().replace("_", " ")


def extract_reference_label_set(metadata: Optional[Dict[str, Any]]) -> Set[str]:
    """
    Extract verified dataset finding labels from reference case metadata.
    Never infers labels from filenames, notes, or heuristics.
    Returns normalized disease string set.
    """
    if not metadata or not isinstance(metadata, dict):
        return set()

    raw_findings = metadata.get("findings")
    if not raw_findings:
        labels_str = metadata.get("finding_labels")
        if labels_str:
            raw_findings = [p.strip() for p in str(labels_str).replace("|", ",").split(",") if p.strip()]
        else:
            raw_findings = []

    return {normalize_disease_name(f) for f in raw_findings if str(f).strip()}


def compute_similarity_weighted_support(
    matches: List[Dict[str, Any]],
    canonical_diseases: Sequence[str] = CANONICAL_DISEASES,
    global_prevalences: Optional[Dict[str, float]] = None,
    global_frequency_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute similarity-weighted disease label support across Top-K retrieved reference cases.

    Mathematical Formulation:
        Let K be the number of retrieved reference cases, each with similarity score S_i in [0.0, 100.0].
        Non-negative weight for case i: w_i = max(0.0, S_i)
        Total weight: W = sum_{i=1}^K w_i

        For disease d in canonical classes:
            Indicator: 1(d in Y_i)
            Weighted Support: V_d = (sum_{i=1}^K w_i * 1(d in Y_i) / W) * 100.0   (if W > 0 else None / 0.0)
            Retrieved Count:  K_d = sum_{i=1}^K 1(d in Y_i)
            Retrieval Rate:   R_d = (K_d / K) * 100.0   (if K > 0 else 0.0)

    Args:
        matches: List of match dictionaries returned by rank_similar_images.
        canonical_diseases: Sequence of disease names (defaults to 14 canonical classes).
        global_prevalences: Optional dict mapping disease name -> dataset prevalence percentage P_d.
        global_frequency_data: Optional full dictionary from disease_frequency.json with counts and denominator.

    Returns:
        List of dicts for each disease, sorted by weighted_support descending:
        [
            {
                "disease": "Pneumonia",
                "disease_name": "Pneumonia",
                "weighted_support": 42.15,
                "retrieved_count": 2,
                "retrieval_rate": 40.0,
                "retrieved_denominator": 5,
                "dataset_prevalence": 1.2763,
                "dataset_occurrence_count": 1431,
                "dataset_denominator": 112120,
                "is_available": True,
                "reason": None,
                "top_k": 5,
            },
            ...
        ]
    """
    k = len(matches)

    # Global prevalence map and counts resolution
    prev_map: Dict[str, float] = {}
    count_map: Dict[str, int] = {}
    dataset_denom: Optional[int] = None

    if global_frequency_data and isinstance(global_frequency_data, dict):
        dataset_denom = global_frequency_data.get("denominator") or global_frequency_data.get("total_valid_records")
        for rec in global_frequency_data.get("diseases", []):
            d_name = rec.get("disease_name")
            if d_name:
                prev_map[d_name] = float(rec.get("prevalence_percentage", 0.0))
                prev_map[normalize_disease_name(d_name)] = float(rec.get("prevalence_percentage", 0.0))
                count_map[d_name] = int(rec.get("occurrence_count", 0))
                count_map[normalize_disease_name(d_name)] = int(rec.get("occurrence_count", 0))

    if global_prevalences and isinstance(global_prevalences, dict):
        for k_prev, v_prev in global_prevalences.items():
            if v_prev is not None:
                prev_map[k_prev] = float(v_prev)
                prev_map[normalize_disease_name(k_prev)] = float(v_prev)

    # Boundary case: K == 0 (no cases retrieved)
    if k == 0:
        return [
            {
                "disease": d,
                "disease_name": d,
                "weighted_support": 0.0,
                "retrieved_count": 0,
                "retrieval_rate": 0.0,
                "retrieved_denominator": 0,
                "dataset_prevalence": prev_map.get(d, prev_map.get(normalize_disease_name(d))),
                "dataset_occurrence_count": count_map.get(d, count_map.get(normalize_disease_name(d))),
                "dataset_denominator": dataset_denom,
                "is_available": False,
                "reason": "No reference cases retrieved",
                "top_k": 0,
            }
            for d in canonical_diseases
        ]

    # Pre-extract weights and findings sets
    weights: List[float] = []
    case_findings: List[Set[str]] = []

    for m in matches:
        score = float(m.get("similarity_score", 0.0))
        w = max(0.0, score)
        weights.append(w)

        meta = m.get("metadata")
        normalized = extract_reference_label_set(meta)
        case_findings.append(normalized)

    total_weight = sum(weights)
    is_weight_zero = total_weight <= 0.0

    support_records: List[Dict[str, Any]] = []
    for disease in canonical_diseases:
        norm_d = normalize_disease_name(disease)
        count = sum(1 for f_set in case_findings if norm_d in f_set)
        retrieval_rate = round((count / k) * 100.0, 2)

        prev = prev_map.get(disease, prev_map.get(norm_d))
        occ_cnt = count_map.get(disease, count_map.get(norm_d))

        if is_weight_zero:
            support_records.append({
                "disease": disease,
                "disease_name": disease,
                "weighted_support": 0.0,
                "retrieved_count": count,
                "retrieval_rate": retrieval_rate,
                "retrieved_denominator": k,
                "dataset_prevalence": prev,
                "dataset_occurrence_count": occ_cnt,
                "dataset_denominator": dataset_denom,
                "is_available": False,
                "reason": "Zero total similarity weight across retrieved cases",
                "top_k": k,
            })
        else:
            weighted_sum = sum(w for w, f_set in zip(weights, case_findings) if norm_d in f_set)
            v_d = round((weighted_sum / total_weight) * 100.0, 2)
            support_records.append({
                "disease": disease,
                "disease_name": disease,
                "weighted_support": v_d,
                "retrieved_count": count,
                "retrieval_rate": retrieval_rate,
                "retrieved_denominator": k,
                "dataset_prevalence": prev,
                "dataset_occurrence_count": occ_cnt,
                "dataset_denominator": dataset_denom,
                "is_available": True,
                "reason": None,
                "top_k": k,
            })

    # Sort primarily by weighted_support descending, then retrieval_rate descending
    support_records.sort(
        key=lambda rec: (rec["weighted_support"] if rec["weighted_support"] is not None else -1.0, rec["retrieval_rate"]),
        reverse=True,
    )
    return support_records


def compute_predicted_findings_evidence(
    matches: List[Dict[str, Any]],
    predicted_findings: Sequence[str],
    global_prevalences: Optional[Dict[str, float]] = None,
    global_frequency_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute prediction-driven, disease-specific reference evidence for each predicted finding.

    For each predicted finding d:
        - matching_cases: number of retrieved reference cases whose dataset labels include d.
        - retrieved_cases: number of eligible cases actually retrieved (K).
        - matching_rate: (matching_cases / retrieved_cases) * 100.0.
        - weighted_support: similarity-weighted label support calculated using similarity weights.
        - is_matched: boolean flag indicating if matching_cases > 0.
        - matching_references: detailed records of matching retrieved cases for case inspection.

    Args:
        matches: List of match dictionaries returned by rank_similar_images.
        predicted_findings: Sequence of disease names predicted for the query radiograph.
        global_prevalences: Optional mapping of disease name -> NIH dataset prevalence percentage.
        global_frequency_data: Optional full dictionary from disease_frequency.json.

    Returns:
        List of evidence records, one for each predicted finding.
    """
    k = len(matches)

    # Resolve global prevalence map for baseline comparison
    prev_map: Dict[str, float] = {}
    if global_frequency_data and isinstance(global_frequency_data, dict):
        for rec in global_frequency_data.get("diseases", []):
            d_name = rec.get("disease_name")
            if d_name:
                prev_map[d_name] = float(rec.get("prevalence_percentage", 0.0))
                prev_map[normalize_disease_name(d_name)] = float(rec.get("prevalence_percentage", 0.0))
        if "no_finding" in global_frequency_data:
            nf = global_frequency_data["no_finding"]
            prev_map["No Finding"] = float(nf.get("prevalence_percentage", 0.0))
            prev_map["no finding"] = float(nf.get("prevalence_percentage", 0.0))

    if global_prevalences and isinstance(global_prevalences, dict):
        for k_prev, v_prev in global_prevalences.items():
            if v_prev is not None:
                prev_map[k_prev] = float(v_prev)
                prev_map[normalize_disease_name(k_prev)] = float(v_prev)

    # Normalize list of predicted findings, filtering empty strings
    clean_findings: List[str] = [str(f).strip() for f in predicted_findings if str(f).strip()]
    if not clean_findings:
        clean_findings = [NO_FINDING_LABEL]

    # Pre-extract weights and findings sets from matches
    weights: List[float] = []
    case_findings: List[Set[str]] = []
    for m in matches:
        score = float(m.get("similarity_score", 0.0))
        weights.append(max(0.0, score))
        meta = m.get("metadata")
        normalized_labels = extract_reference_label_set(meta)
        if not normalized_labels:
            raw_f = m.get("findings") or []
            normalized_labels = {normalize_disease_name(x) for x in raw_f if str(x).strip()}
        case_findings.append(normalized_labels)

    total_weight = sum(weights)
    is_weight_zero = total_weight <= 0.0

    evidence_records: List[Dict[str, Any]] = []
    for finding in clean_findings:
        norm_finding = normalize_disease_name(finding)
        is_normal_finding = norm_finding in ("no finding", "normal")

        # Identify which matches contain the predicted finding
        matching_indices: List[int] = []
        for idx, f_set in enumerate(case_findings):
            if is_normal_finding:
                if any(f in ("no finding", "normal") for f in f_set) or len(f_set) == 0:
                    matching_indices.append(idx)
            else:
                if norm_finding in f_set:
                    matching_indices.append(idx)

        matching_cases = len(matching_indices)
        matching_rate = round((matching_cases / k) * 100.0, 1) if k > 0 else 0.0

        if is_weight_zero or k == 0 or matching_cases == 0:
            weighted_support = 0.0
        else:
            weighted_sum = sum(weights[idx] for idx in matching_indices)
            weighted_support = round((weighted_sum / total_weight) * 100.0, 1)

        dataset_prev = prev_map.get(finding, prev_map.get(norm_finding))

        matching_references: List[Dict[str, Any]] = []
        for idx in matching_indices:
            m = matches[idx]
            meta = m.get("metadata") or {}
            age = meta.get("patient_age")
            gender = meta.get("patient_gender")
            view = meta.get("view_position")
            demographics_parts = [f"{age}y" if age else None, gender if gender and gender != "U" else None, view]
            demographics_str = " · ".join(p for p in demographics_parts if p)

            matching_references.append({
                "rank": m.get("rank", idx + 1),
                "filename": m.get("filename", ""),
                "image_id": m.get("image_id", m.get("filename", "")),
                "similarity_score": m.get("similarity_score", 0.0),
                "findings": m.get("findings", []),
                "finding_labels": m.get("finding_labels", ""),
                "patient_age": age,
                "patient_gender": gender,
                "view_position": view,
                "demographics": demographics_str or "N/A",
                "dataset_source": m.get("provenance", meta.get("dataset_source", "NIH ChestX-ray14")),
                "is_synthetic": m.get("is_synthetic", False),
            })

        is_matched = matching_cases > 0

        if k == 0:
            explanation = f"0 of 0 retrieved reference cases (0.0% matching rate, 0.0% weighted support) match predicted {finding}."
        elif matching_cases == 0:
            explanation = (
                f"0 of {k} retrieved reference cases (0.0% matching rate, 0.0% weighted support) match predicted {finding}. "
                f"No reference support found in top matches."
            )
        else:
            explanation = (
                f"{matching_cases} of {k} retrieved reference cases ({matching_rate:.1f}% matching rate, "
                f"{weighted_support:.1f}% weighted support) corroborate predicted {finding}."
            )

        evidence_records.append({
            "predicted_finding": finding,
            "finding": finding,
            "disease": finding,
            "disease_name": finding,
            "matching_cases": matching_cases,
            "retrieved_cases": k,
            "matching_rate": matching_rate,
            "weighted_support": weighted_support,
            "dataset_prevalence": dataset_prev,
            "is_matched": is_matched,
            "matching_references": matching_references,
            "explanation": explanation,
        })

    return evidence_records


def evaluate_diagnostic_concordance(
    reference_metadata: Optional[Dict[str, Any]],
    query_findings: Optional[List[str]] = None,
    primary_finding: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate clinical and diagnostic concordance between query model predictions
    and verified ground-truth findings of a reference case.

    Concordance Categories:
    - PRIMARY_CONCORDANT: Reference case shares the primary detected pathology.
    - SECONDARY_OVERLAP: Reference case shares secondary detected pathology, but not the primary.
    - DIFFERENTIAL: Reference case exhibits visual similarity with differing verified pathology.
    - NO_FINDING: Both query and reference indicate No Finding / Normal.
    - NEUTRAL: Uninformative or uncorrelated presentation (e.g. metadata unavailable).
    """
    if not reference_metadata:
        return {
            "status": "NEUTRAL",
            "label": "Metadata Unavailable",
            "matches_primary": False,
            "has_overlap": False,
            "is_concordant": False,
            "shared_findings": [],
            "reference_findings": [],
            "description": "Clinical metadata is not recorded for this reference case.",
        }

    raw_ref_findings = reference_metadata.get("findings")
    if not raw_ref_findings:
        labels = reference_metadata.get("finding_labels", "No Finding")
        raw_ref_findings = [p.strip() for p in str(labels).replace("|", ",").split(",") if p.strip()]

    ref_findings = [str(f).strip() for f in raw_ref_findings if str(f).strip()]
    normalized_ref = {normalize_disease_name(f): f for f in ref_findings}

    norm_query_findings = [normalize_disease_name(f) for f in (query_findings or []) if str(f).strip()]
    norm_primary = normalize_disease_name(primary_finding) if primary_finding else None

    # Check primary match
    matches_primary = bool(norm_primary and norm_primary in normalized_ref and norm_primary not in ("no finding", "normal"))

    # Shared findings
    shared = [normalized_ref[nq] for nq in norm_query_findings if nq in normalized_ref and nq not in ("no finding", "normal")]
    if matches_primary and norm_primary and normalized_ref[norm_primary] not in shared:
        shared.append(normalized_ref[norm_primary])

    # No finding checks
    is_ref_no_finding = all(normalize_disease_name(f) in ("no finding", "normal") for f in ref_findings) or len(ref_findings) == 0
    is_query_no_finding = len(norm_query_findings) == 0 or all(q in ("no finding", "normal") for q in norm_query_findings)

    has_overlap = len(shared) > 0 or matches_primary

    if matches_primary:
        status = "PRIMARY_CONCORDANT"
        label = "Primary Concordant"
        description = f"Reference case shares primary detected finding: {normalized_ref.get(norm_primary, primary_finding)}."
    elif len(shared) > 0:
        status = "SECONDARY_OVERLAP"
        label = "Secondary Overlap"
        findings_str = ", ".join(shared)
        description = f"Reference case shares secondary finding(s): {findings_str}."
    elif is_ref_no_finding and is_query_no_finding:
        status = "NO_FINDING"
        label = "Normal / No Finding"
        description = "Both query evaluation and reference case show no focal positive findings."
    elif not is_ref_no_finding and not is_query_no_finding:
        status = "DIFFERENTIAL"
        label = "Differential Lookalike"
        ref_str = ", ".join(ref_findings)
        query_str = primary_finding or (query_findings[0] if query_findings else "predicted pathology")
        description = f"Visual morphological similarity with differing verified pathology ({ref_str} vs predicted {query_str}). Useful as differential lookalike."
    else:
        status = "NEUTRAL"
        label = "Neutral Finding"
        description = "Uncorrelated clinical findings between query and reference."

    return {
        "status": status,
        "label": label,
        "matches_primary": matches_primary,
        "has_overlap": has_overlap,
        "is_concordant": has_overlap,
        "shared_findings": shared,
        "reference_findings": ref_findings,
        "description": description,
    }


def rank_similar_images(
    query_vector: np.ndarray,
    reference_filenames: List[str],
    reference_embeddings: np.ndarray,
    top_k: int = 5,
    base_url: str = "/reference_images/",
    metadata: Optional[List[Dict[str, Any]]] = None,
    query_findings: Optional[List[str]] = None,
    primary_finding: Optional[str] = None,
    exclude_filenames: Optional[Union[str, Sequence[str]]] = None,
    authentic_only: bool = False,
    exclude_identical_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """
    Rank reference images against the query vector and return Top-K matches.

    Args:
        query_vector: Feature vector for the query radiograph.
        reference_filenames: List of filenames for the reference images.
        reference_embeddings: Matrix of precomputed feature vectors (N, D).
        top_k: Number of most similar images to return (default: 5).
        base_url: URL or path prefix for referencing image assets.
        metadata: Optional list of dictionaries with clinical or demographic metadata.
        query_findings: Optional list of detected positive findings for the query image.
        primary_finding: Optional primary disease detected for the query image.
        exclude_filenames: Optional filename or sequence of filenames to exclude from retrieval.
        authentic_only: If True, exclude synthetic benchmark/demonstration cases from retrieval candidates.
        exclude_identical_embeddings: If True, exclude candidates with cosine similarity >= 99.99% (self-retrieval).

    Returns:
        List of dictionaries sorted by similarity_score in descending order.
    """
    if len(reference_filenames) == 0 or len(reference_embeddings) == 0:
        return []

    if len(reference_filenames) != len(reference_embeddings):
        raise ValueError(
            f"Mismatch: {len(reference_filenames)} filenames vs {len(reference_embeddings)} embeddings."
        )

    excluded_set: set[str] = set()
    if exclude_filenames:
        if isinstance(exclude_filenames, str):
            excluded_set.add(exclude_filenames.strip().lower())
            excluded_set.add(Path(exclude_filenames).name.lower())
        else:
            for f in exclude_filenames:
                excluded_set.add(str(f).strip().lower())
                excluded_set.add(Path(str(f)).name.lower())

    candidate_indices: List[int] = []
    for i, fn in enumerate(reference_filenames):
        fn_str = str(fn).strip().lower()
        if fn_str in excluded_set or Path(fn_str).name in excluded_set:
            continue
        item_meta = metadata[i] if metadata is not None and i < len(metadata) else None
        if authentic_only:
            src = str(item_meta.get("dataset_source", "")).strip().lower() if item_meta else ""
            if "synthetic" in src or "demo" in src or fn_str.startswith("xray00"):
                continue
        candidate_indices.append(i)

    if len(candidate_indices) == 0:
        return []

    sub_embeddings = reference_embeddings[candidate_indices]
    sub_filenames = [reference_filenames[i] for i in candidate_indices]
    sub_metadata = [metadata[i] for i in candidate_indices] if metadata is not None else None

    scores = compute_cosine_similarity(query_vector, sub_embeddings)

    if exclude_identical_embeddings:
        non_identical = [idx for idx, s in enumerate(scores) if s < 99.99]
        if len(non_identical) > 0 and len(non_identical) < len(scores):
            sub_embeddings = sub_embeddings[non_identical]
            sub_filenames = [sub_filenames[idx] for idx in non_identical]
            sub_metadata = [sub_metadata[idx] for idx in non_identical] if sub_metadata is not None else None
            scores = scores[non_identical]

    # Sort indices in descending order of similarity
    sorted_sub_indices = np.argsort(scores)[::-1]

    k = min(max(1, top_k), len(sorted_sub_indices))
    selected_indices = sorted_sub_indices[:k]

    results: List[Dict[str, Any]] = []
    for rank_idx, idx in enumerate(selected_indices, start=1):
        filename = str(sub_filenames[idx])
        score = round(float(scores[idx]), 2)

        item_meta: Optional[Dict[str, Any]] = None
        if sub_metadata is not None and idx < len(sub_metadata):
            item_meta = sub_metadata[idx]

        # Extract image identifier and provenance
        image_id = filename
        provenance = "NIH ChestX-ray14"
        is_synthetic = False
        if item_meta:
            image_id = str(item_meta.get("image_index") or item_meta.get("Image Index") or filename)
            src = str(item_meta.get("dataset_source", "")).strip()
            if src:
                provenance = src
            if "synthetic" in provenance.lower() or "demo" in image_id.lower() or filename.startswith("xray00"):
                is_synthetic = True

        raw_findings_list: List[str] = []
        if item_meta:
            raw_findings = item_meta.get("findings")
            if raw_findings:
                raw_findings_list = [str(f).strip() for f in raw_findings if str(f).strip()]
            else:
                lbls = item_meta.get("finding_labels")
                if lbls:
                    raw_findings_list = [p.strip() for p in str(lbls).replace("|", ",").split(",") if p.strip()]

        item: Dict[str, Any] = {
            "rank": rank_idx,
            "filename": filename,
            "image_id": image_id,
            "original_image_identifier": image_id,
            "similarity_score": score,
            "image_url": f"{base_url.rstrip('/')}/{filename}",
            "findings": raw_findings_list,
            "finding_labels": "|".join(raw_findings_list) if raw_findings_list else "No Finding",
            "provenance": provenance,
            "is_synthetic": is_synthetic,
        }

        if item_meta is not None:
            item["metadata"] = item_meta

        # Evaluate diagnostic concordance
        if query_findings is not None or primary_finding is not None:
            item["concordance"] = evaluate_diagnostic_concordance(
                reference_metadata=item_meta,
                query_findings=query_findings,
                primary_finding=primary_finding,
            )

        results.append(item)

    return results
