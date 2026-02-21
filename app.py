"""
Streamlit UI for Chemical-Disease Link Prediction.

Provides three prediction modes:
1. Predict association between a disease and a chemical
2. Get top-k chemicals for a disease
3. Get top-k diseases for a chemical
"""

import streamlit as st
import torch
import polars as pl
from pathlib import Path

from src.data.processing import load_processed_data
from src.data.graph import build_graph_from_processed
from src.models.inference.cached_embeddings import CachedEmbeddingPredictor
from src.explainability.explain import build_node_names, ExplanationResult


@st.cache_resource
def load_model_and_data(
    processed_dir: str = './data/processed',
    embeddings_dir: str = './embeddings',
    threshold: float = 0.5
):
    """Load and cache metadata + cached-embedding predictor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_dict = load_processed_data(processed_dir)
    cache_path = Path(embeddings_dir)
    required_files = [
        cache_path / 'chemical_embeddings.npy',
        cache_path / 'disease_embeddings.npy',
        cache_path / 'W_cd.pt'
    ]
    missing = [str(p.name) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cached embeddings are required for Streamlit inference to avoid OOM. "
            f"Missing files in {embeddings_dir}: {missing}. "
            "Run scripts/cache_embeddings_chunked.py first."
        )

    predictor = CachedEmbeddingPredictor.from_cache(
        cache_dir=embeddings_dir,
        disease_df=data_dict['diseases'],
        chemical_df=data_dict['chemicals'],
        chem_disease_df=data_dict.get('chem_disease'),
        device=device,
        threshold=threshold
    )
    return predictor, data_dict


@st.cache_resource
def load_graph_for_explain(
    processed_dir: str = './data/processed',
):
    """Load the HeteroData graph for explainability (cached)."""
    data, _ = build_graph_from_processed(
        processed_data_dir=processed_dir,
        add_reverse_edges=True,
        save_vocabs=False,
        include_extended=True,
    )
    return data


def _render_explanation(explanation: ExplanationResult):
    """Render an ExplanationResult in the Streamlit UI."""
    if not explanation.paths:
        st.info("No explanatory paths found between this chemical and disease.")
        return

    st.markdown(
        f"**{len(explanation.paths)} paths found** | "
        f"Attention scoring: {'Yes' if explanation.attention_available else 'No (Tier 1 only)'}"
    )

    # Metapath summary chips
    if explanation.metapath_summary:
        cols = st.columns(min(len(explanation.metapath_summary), 4))
        for i, (mp_type, count) in enumerate(
            sorted(explanation.metapath_summary.items(), key=lambda x: -x[1])
        ):
            cols[i % len(cols)].metric(mp_type, count)

    # Top paths table
    rows = []
    for i, sp in enumerate(explanation.top_paths, 1):
        rows.append({
            "Rank": i,
            "Evidence": sp.evidence_type,
            "Score": f"{sp.combined_score:.4f}",
            "Attention": f"{sp.attention_score:.4f}",
            "Embedding": f"{sp.embedding_score:.4f}",
            "Path": sp.description,
        })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Evidence": st.column_config.TextColumn("Evidence Type"),
                "Score": st.column_config.TextColumn("Score"),
                "Attention": st.column_config.TextColumn("Attn"),
                "Embedding": st.column_config.TextColumn("Emb"),
                "Path": st.column_config.TextColumn("Path", width="large"),
            },
        )


def _resolve_id(selection: str, options: dict, manual_id: str, valid_ids: set) -> str:
    if manual_id:
        manual_id = manual_id.strip()
        return manual_id if manual_id in valid_ids else ""
    return options.get(selection, "") if selection else ""


def _example_caption(ids: list, names: list, count: int = 3) -> str:
    examples = []
    for idx, name in list(zip(ids, names))[:count]:
        examples.append(f"{idx} ({name})")
    return "Examples: " + "; ".join(examples) if examples else ""


def _build_known_pair_examples(
    data_dict: dict,
    disease_name_map: dict,
    chemical_name_map: dict,
    max_pairs: int = 5
) -> list:
    if 'chem_disease' not in data_dict:
        return []
    diseases = data_dict['diseases']
    chemicals = data_dict['chemicals']
    disease_id_map = dict(zip(diseases['DS_ID'].to_list(), diseases['DS_OMIM_MESH_ID'].to_list()))
    chem_id_map = dict(zip(chemicals['CHEM_ID'].to_list(), chemicals['CHEM_MESH_ID'].to_list()))
    pairs = []
    for row in data_dict['chem_disease'].select(['CHEM_ID', 'DS_ID']).iter_rows(named=True):
        disease_id = disease_id_map.get(row['DS_ID'])
        chemical_id = chem_id_map.get(row['CHEM_ID'])
        if not disease_id or not chemical_id:
            continue
        pairs.append({
            'disease_id': disease_id,
            'disease_name': disease_name_map.get(disease_id, 'Unknown'),
            'chemical_id': chemical_id,
            'chemical_name': chemical_name_map.get(chemical_id, 'Unknown')
        })
        if len(pairs) >= max_pairs:
            break
    return pairs


def _find_examples_with_known(
    predictor,
    ids: list,
    mode: str,
    min_known: int = 2,
    max_known: int = 3,
    top_k: int = 50,
    max_scan: int = 120
) -> list:
    examples = []
    for item_id in ids[:max_scan]:
        try:
            if mode == 'disease':
                results_df = predictor.predict_chemicals_for_disease(
                    item_id, top_k=top_k, exclude_known=False
                )
            else:
                results_df = predictor.predict_diseases_for_chemical(
                    item_id, top_k=top_k, exclude_known=False
                )
        except Exception:
            continue
        known_count = results_df.filter(pl.col('known') == True).height
        if min_known <= known_count <= max_known:
            examples.append({
                'id': item_id,
                'known_count': known_count,
                'top_k': top_k
            })
        if len(examples) >= 5:
            break
    return examples


def _set_state(updates: dict) -> None:
    for key, value in updates.items():
        st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="Chemical-Disease Link Prediction",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("Chemical-Disease Link Prediction")
    st.markdown("Predict associations between chemicals and diseases using a heterogeneous graph neural network.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        processed_dir = st.text_input("Processed Data Directory", value="./data/processed")
        embeddings_dir = st.text_input("Embeddings Directory", value="./embeddings")
        
        threshold = st.slider("Classification Threshold", value=0.5, min_value=0.0, max_value=1.0)
        st.caption("Inference runs in cached-embedding mode to reduce memory usage.")
    
    # Load model
    try:
        with st.spinner("Loading model and data..."):
            predictor, data_dict = load_model_and_data(
                processed_dir=processed_dir,
                embeddings_dir=embeddings_dir,
                threshold=threshold
            )
            predictor.threshold = threshold
        st.sidebar.success("Loaded cached embeddings successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Get available IDs for autocomplete
    disease_ids = data_dict['diseases']['DS_OMIM_MESH_ID'].to_list()
    disease_names = data_dict['diseases']['DS_NAME'].to_list()
    disease_options = {f"{did} - {dname}": did for did, dname in zip(disease_ids, disease_names)}
    disease_id_set = set(disease_ids)
    
    chemical_ids = data_dict['chemicals']['CHEM_MESH_ID'].to_list()
    chemical_names = data_dict['chemicals']['CHEM_NAME'].to_list()
    chemical_options = {f"{cid} - {cname}": cid for cid, cname in zip(chemical_ids, chemical_names)}
    chemical_id_set = set(chemical_ids)
    disease_name_map = dict(zip(disease_ids, disease_names))
    chemical_name_map = dict(zip(chemical_ids, chemical_names))
    disease_examples = _example_caption(disease_ids, disease_names)
    chemical_examples = _example_caption(chemical_ids, chemical_names)
    known_pair_examples = _build_known_pair_examples(
        data_dict, disease_name_map, chemical_name_map, max_pairs=5
    )

    demo_pair_examples = [
        {
            'disease_id': 'MESH:D014202',
            'disease_name': 'Tremor',
            'chemical_id': 'C006901',
            'chemical_name': 'carbophenothion'
        },
        {
            'disease_id': 'MESH:D020767',
            'disease_name': 'Intracranial Thrombosis',
            'chemical_id': 'D005343',
            'chemical_name': 'Fibrinolytic Agents'
        },
        {
            'disease_id': 'MESH:D006083',
            'disease_name': 'Graft Occlusion, Vascular',
            'chemical_id': 'D017984',
            'chemical_name': 'Enoxaparin'
        },
        {
            'disease_id': 'MESH:D004342',
            'disease_name': 'Drug Hypersensitivity',
            'chemical_id': 'D002515',
            'chemical_name': 'Cephradine'
        },
        {
            'disease_id': 'MESH:D043183',
            'disease_name': 'Irritable Bowel Syndrome',
            'chemical_id': 'D000068238',
            'chemical_name': 'Lubiprostone'
        }
    ]

    demo_disease_examples = [
        {
            'id': 'MESH:D014202',
            'name': 'Tremor',
            'known_count': 3,
            'top_k': 10
        },
        {
            'id': 'MESH:D020767',
            'name': 'Intracranial Thrombosis',
            'known_count': 1,
            'top_k': 10
        },
        {
            'id': 'MESH:D006083',
            'name': 'Graft Occlusion, Vascular',
            'known_count': 1,
            'top_k': 10
        },
        {
            'id': 'MESH:D004342',
            'name': 'Drug Hypersensitivity',
            'known_count': 2,
            'top_k': 10
        },
        {
            'id': 'MESH:D043183',
            'name': 'Irritable Bowel Syndrome',
            'known_count': 1,
            'top_k': 10
        }
    ]

    demo_chemical_examples = [
        {
            'id': 'C054074',
            'name': 'niguldipine',
            'known_count': 2,
            'top_k': 10
        },
        {
            'id': 'D008879',
            'name': 'Midodrine',
            'known_count': 1,
            'top_k': 10
        },
        {
            'id': 'C079148',
            'name': '(3-aminopropyl)(n-butyl)phosphinic acid',
            'known_count': 1,
            'top_k': 10
        },
        {
            'id': 'D017312',
            'name': 'Toremifene',
            'known_count': 2,
            'top_k': 10
        },
        {
            'id': 'C041259',
            'name': 'Disperse Orange 3',
            'known_count': 1,
            'top_k': 10
        }
    ]

    # Sidebar data summary
    with st.sidebar:
        with st.expander("Dataset summary", expanded=False):
            st.write(f"Diseases: {len(disease_ids):,}")
            st.write(f"Chemicals: {len(chemical_ids):,}")
            if 'genes' in data_dict:
                st.write(f"Genes: {data_dict['genes'].height:,}")
            if 'pathways' in data_dict:
                st.write(f"Pathways: {data_dict['pathways'].height:,}")
            if 'go_terms' in data_dict:
                st.write(f"GO terms: {data_dict['go_terms'].height:,}")
            if 'chem_disease' in data_dict:
                st.write(f"Known chem-disease links: {data_dict['chem_disease'].height:,}")
    
    # Create tabs for the three prediction modes
    tab1, tab2, tab3 = st.tabs([
        "Pair Prediction",
        "Top Chemicals for Disease", 
        "Top Diseases for Chemical"
    ])
    
    # Tab 1: Pair Prediction
    with tab1:
        st.header("Predict Disease-Chemical Association")
        st.markdown("Enter a disease and chemical ID to predict their association probability.")
        if demo_pair_examples:
            st.subheader("Demo pairs (CTD-known)")
            cols = st.columns(5)
            for idx, ex in enumerate(demo_pair_examples):
                label = f"{ex['disease_id']} + {ex['chemical_id']}"
                if cols[idx].button(label, key=f"demo_pair_{idx}"):
                    _set_state({
                        'pair_disease_manual': ex['disease_id'],
                        'pair_chemical_manual': ex['chemical_id'],
                        'pair_disease': "",
                        'pair_chemical': ""
                    })
                cols[idx].caption(f"{ex['disease_name']} • {ex['chemical_name']}")
        elif known_pair_examples:
            with st.expander("CTD-known example pairs", expanded=False):
                for ex in known_pair_examples:
                    st.write(
                        f"{ex['disease_id']} ({ex['disease_name']}) + "
                        f"{ex['chemical_id']} ({ex['chemical_name']})"
                    )
        
        col1, col2 = st.columns(2)
        
        with col1:
            disease_selection = st.selectbox(
                "Select Disease (MESH/OMIM ID)",
                options=[""] + list(disease_options.keys()),
                key="pair_disease",
                help="Search by ID or name"
            )
            if disease_examples:
                st.caption(disease_examples)
            disease_manual = st.text_input(
                "Or enter Disease ID",
                value="",
                key="pair_disease_manual",
                placeholder="MESH:D003920"
            )
            disease_id = _resolve_id(disease_selection, disease_options, disease_manual, disease_id_set)
            if disease_id:
                st.caption(f"Selected disease: {disease_name_map.get(disease_id, 'Unknown')}")
        
        with col2:
            chemical_selection = st.selectbox(
                "Select Chemical (MESH ID)",
                options=[""] + list(chemical_options.keys()),
                key="pair_chemical",
                help="Search by ID or name"
            )
            if chemical_examples:
                st.caption(chemical_examples)
            chemical_manual = st.text_input(
                "Or enter Chemical ID",
                value="",
                key="pair_chemical_manual",
                placeholder="D015525"
            )
            chemical_id = _resolve_id(chemical_selection, chemical_options, chemical_manual, chemical_id_set)
            if chemical_id:
                st.caption(f"Selected chemical: {chemical_name_map.get(chemical_id, 'Unknown')}")
        
        if disease_manual and not disease_id:
            st.warning("Unknown disease ID. Use a valid MESH/OMIM ID from the dataset.")
        if chemical_manual and not chemical_id:
            st.warning("Unknown chemical ID. Use a valid MESH ID from the dataset.")
        
        if st.button("Predict Association", key="predict_pair"):
            if not disease_id or not chemical_id:
                st.warning("Please select both a disease and a chemical.")
            else:
                try:
                    with st.spinner("Computing prediction..."):
                        result = predictor.predict_pair(disease_id, chemical_id)
                    st.session_state.pair_result = result
                    st.session_state.pair_explanation = None
                except ValueError as e:
                    st.error(f"Error: {e}")

        # Display persisted prediction result
        if st.session_state.get("pair_result") is not None:
            result = st.session_state.pair_result
            # Clear stale results when inputs change
            if result['disease_id'] != disease_id or result['chemical_id'] != chemical_id:
                st.session_state.pair_result = None
                st.session_state.pair_explanation = None
            else:
                st.subheader("Prediction Result")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probability", f"{result['probability']:.4f}")
                with col2:
                    label_text = "Associated" if result['label'] == 1 else "Not Associated"
                    st.metric("Prediction", label_text)
                with col3:
                    known_text = "Yes" if result.get('known', False) else "No"
                    st.metric("Known Association", known_text)

                with st.expander("Details"):
                    st.write(f"**Disease:** {result['disease_name']} ({result['disease_id']})")
                    st.write(f"**Chemical:** {result['chemical_name']} ({result['chemical_id']})")
                    st.write(f"**Logit:** {result['logit']:.4f}")
                    st.write(f"**Threshold:** {threshold}")
                    st.write(f"**Known in CTD:** {'Yes' if result.get('known', False) else 'No'}")
                    st.write("**Mode:** Cached embeddings")

                # Explainability
                if st.button("Explain prediction", key="explain_pair"):
                    try:
                        with st.spinner("Generating explanation (enumerating paths)..."):
                            node_names = build_node_names(data_dict)
                            graph = load_graph_for_explain(
                                processed_dir=processed_dir,
                            )
                            explanation = predictor.explain_prediction(
                                disease_id, chemical_id,
                                data=graph,
                                node_names=node_names,
                            )
                        st.session_state.pair_explanation = explanation
                    except Exception as ex:
                        st.error(f"Explanation error: {ex}")

                if st.session_state.get("pair_explanation") is not None:
                    _render_explanation(st.session_state.pair_explanation)
    
    # Tab 2: Top Chemicals for Disease
    with tab2:
        st.header("Top Chemicals for Disease")
        st.markdown("Find the top-k chemicals most likely associated with a disease.")
        if demo_disease_examples:
            st.subheader("Demo diseases (top-10 has known + novel)")
            cols = st.columns(5)
            for idx, ex in enumerate(demo_disease_examples):
                label = ex['id']
                if cols[idx].button(label, key=f"demo_disease_{idx}"):
                    _set_state({
                        'topk_disease_manual': ex['id'],
                        'topk_disease_select': ""
                    })
                cols[idx].caption(f"{ex['name']} | known {ex['known_count']} in top-{ex['top_k']}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            disease_selection_2 = st.selectbox(
                "Select Disease (MESH/OMIM ID)",
                options=[""] + list(disease_options.keys()),
                key="topk_disease_select",
                help="Search by ID or name"
            )
            if disease_examples:
                st.caption(disease_examples)
            disease_manual_2 = st.text_input(
                "Or enter Disease ID",
                value="",
                key="topk_disease_manual",
                placeholder="MESH:D003920"
            )
            disease_id_2 = _resolve_id(disease_selection_2, disease_options, disease_manual_2, disease_id_set)
        
        with col2:
            top_k_chem = st.number_input("Top K", value=10, min_value=1, max_value=100, key="topk_chem")
        
        exclude_known_chem = st.checkbox("Exclude known associations", value=True, key="exclude_known_chem")
        known_filter_chem = st.selectbox(
            "Known filter",
            options=["All", "Known only", "Novel only"],
            index=0,
            key="known_filter_chem"
        )

        if exclude_known_chem and known_filter_chem == "Known only":
            st.info("Known-only filter with exclusion enabled will return no results.")
        
        if disease_manual_2 and not disease_id_2:
            st.warning("Unknown disease ID. Use a valid MESH/OMIM ID from the dataset.")
        
        if st.button("Find Top Chemicals", key="find_chemicals"):
            if not disease_id_2:
                st.warning("Please select a disease.")
            else:
                try:
                    with st.spinner("Finding top chemicals..."):
                        results_df = predictor.predict_chemicals_for_disease(
                            disease_id_2,
                            top_k=top_k_chem,
                            exclude_known=exclude_known_chem
                        )
                    if known_filter_chem == "Known only":
                        results_df = results_df.filter(pl.col('known') == True)
                    elif known_filter_chem == "Novel only":
                        results_df = results_df.filter(pl.col('known') == False)
                    st.session_state.tab2_results = results_df
                    st.session_state.tab2_disease_id = disease_id_2
                    st.session_state.tab2_explanations = {}
                except ValueError as e:
                    st.error(f"Error: {e}")

        # Display persisted Tab 2 results
        if st.session_state.get("tab2_results") is not None:
            # Clear stale results when disease input changes
            if st.session_state.get("tab2_disease_id") != disease_id_2:
                st.session_state.tab2_results = None
                st.session_state.tab2_disease_id = None
                st.session_state.tab2_explanations = {}
            else:
                results_df = st.session_state.tab2_results
                if results_df.height == 0:
                    st.info("No results after applying the selected filters.")
                else:
                    st.subheader(f"Top {top_k_chem} Chemicals for {disease_id_2}")
                    st.caption(f"Disease: {disease_name_map.get(disease_id_2, 'Unknown')}")
                    known_count = results_df.filter(pl.col('known') == True).height
                    st.caption(f"Known: {known_count} | Novel: {results_df.height - known_count}")

                    # Convert to pandas for display
                    display_df = results_df.to_pandas()
                    display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.4f}")
                    display_df['logit'] = display_df['logit'].apply(lambda x: f"{x:.4f}")

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank"),
                            "chemical_id": st.column_config.TextColumn("Chemical ID"),
                            "chemical_name": st.column_config.TextColumn("Chemical Name"),
                            "probability": st.column_config.TextColumn("Probability"),
                            "logit": st.column_config.TextColumn("Logit"),
                            "known": st.column_config.CheckboxColumn("Known", help="Known association in CTD database")
                        }
                    )

                    # Download button
                    csv = results_df.write_csv()
                    st.download_button(
                        "Download Results (CSV)",
                        csv,
                        file_name=f"top_chemicals_{disease_id_2.replace(':', '_')}.csv",
                        mime="text/csv"
                    )

                    # Per-result explainability
                    st.subheader("Explain individual results")
                    for row in results_df.iter_rows(named=True):
                        chem_id_row = row['chemical_id']
                        chem_name_row = row['chemical_name']
                        explain_key = f"{disease_id_2}_{chem_id_row}"
                        with st.expander(f"Explain: {chem_name_row} ({chem_id_row})"):
                            if st.button(
                                "Generate explanation",
                                key=f"explain_tab2_{explain_key}",
                            ):
                                try:
                                    with st.spinner("Generating explanation..."):
                                        node_names = build_node_names(data_dict)
                                        graph = load_graph_for_explain(
                                            processed_dir=processed_dir,
                                        )
                                        explanation = predictor.explain_prediction(
                                            disease_id_2, chem_id_row,
                                            data=graph,
                                            node_names=node_names,
                                        )
                                    st.session_state.tab2_explanations[explain_key] = explanation
                                except Exception as ex:
                                    st.error(f"Explanation error: {ex}")

                            if explain_key in st.session_state.get("tab2_explanations", {}):
                                _render_explanation(st.session_state.tab2_explanations[explain_key])
    
    # Tab 3: Top Diseases for Chemical
    with tab3:
        st.header("Top Diseases for Chemical")
        st.markdown("Find the top-k diseases most likely associated with a chemical.")
        if demo_chemical_examples:
            st.subheader("Demo chemicals (top-10 has known + novel)")
            cols = st.columns(5)
            for idx, ex in enumerate(demo_chemical_examples):
                label = ex['id']
                if cols[idx].button(label, key=f"demo_chemical_{idx}"):
                    _set_state({
                        'topk_chemical_manual': ex['id'],
                        'topk_chemical': ""
                    })
                cols[idx].caption(f"{ex['name']} | known {ex['known_count']} in top-{ex['top_k']}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            chemical_selection_2 = st.selectbox(
                "Select Chemical (MESH ID)",
                options=[""] + list(chemical_options.keys()),
                key="topk_chemical",
                help="Search by ID or name"
            )
            if chemical_examples:
                st.caption(chemical_examples)
            chemical_manual_2 = st.text_input(
                "Or enter Chemical ID",
                value="",
                key="topk_chemical_manual",
                placeholder="D015525"
            )
            chemical_id_2 = _resolve_id(chemical_selection_2, chemical_options, chemical_manual_2, chemical_id_set)
        
        with col2:
            top_k_disease = st.number_input("Top K", value=10, min_value=1, max_value=100, key="topk_disease_count")
        
        exclude_known_disease = st.checkbox("Exclude known associations", value=True, key="exclude_known_disease")
        known_filter_disease = st.selectbox(
            "Known filter",
            options=["All", "Known only", "Novel only"],
            index=0,
            key="known_filter_disease"
        )

        if exclude_known_disease and known_filter_disease == "Known only":
            st.info("Known-only filter with exclusion enabled will return no results.")
        
        if chemical_manual_2 and not chemical_id_2:
            st.warning("Unknown chemical ID. Use a valid MESH ID from the dataset.")
        
        if st.button("Find Top Diseases", key="find_diseases"):
            if not chemical_id_2:
                st.warning("Please select a chemical.")
            else:
                try:
                    with st.spinner("Finding top diseases..."):
                        results_df = predictor.predict_diseases_for_chemical(
                            chemical_id_2,
                            top_k=top_k_disease,
                            exclude_known=exclude_known_disease
                        )
                    if known_filter_disease == "Known only":
                        results_df = results_df.filter(pl.col('known') == True)
                    elif known_filter_disease == "Novel only":
                        results_df = results_df.filter(pl.col('known') == False)
                    st.session_state.tab3_results = results_df
                    st.session_state.tab3_chemical_id = chemical_id_2
                    st.session_state.tab3_explanations = {}
                except ValueError as e:
                    st.error(f"Error: {e}")

        # Display persisted Tab 3 results
        if st.session_state.get("tab3_results") is not None:
            # Clear stale results when chemical input changes
            if st.session_state.get("tab3_chemical_id") != chemical_id_2:
                st.session_state.tab3_results = None
                st.session_state.tab3_chemical_id = None
                st.session_state.tab3_explanations = {}
            else:
                results_df = st.session_state.tab3_results
                if results_df.height == 0:
                    st.info("No results after applying the selected filters.")
                else:
                    st.subheader(f"Top {top_k_disease} Diseases for {chemical_id_2}")
                    st.caption(f"Chemical: {chemical_name_map.get(chemical_id_2, 'Unknown')}")
                    known_count = results_df.filter(pl.col('known') == True).height
                    st.caption(f"Known: {known_count} | Novel: {results_df.height - known_count}")

                    # Convert to pandas for display
                    display_df = results_df.to_pandas()
                    display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.4f}")
                    display_df['logit'] = display_df['logit'].apply(lambda x: f"{x:.4f}")

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank"),
                            "disease_id": st.column_config.TextColumn("Disease ID"),
                            "disease_name": st.column_config.TextColumn("Disease Name"),
                            "probability": st.column_config.TextColumn("Probability"),
                            "logit": st.column_config.TextColumn("Logit"),
                            "known": st.column_config.CheckboxColumn("Known", help="Known association in CTD database")
                        }
                    )

                    # Download button
                    csv = results_df.write_csv()
                    st.download_button(
                        "Download Results (CSV)",
                        csv,
                        file_name=f"top_diseases_{chemical_id_2.replace(':', '_')}.csv",
                        mime="text/csv"
                    )

                    # Per-result explainability
                    st.subheader("Explain individual results")
                    for row in results_df.iter_rows(named=True):
                        dis_id_row = row['disease_id']
                        dis_name_row = row['disease_name']
                        explain_key = f"{chemical_id_2}_{dis_id_row}"
                        with st.expander(f"Explain: {dis_name_row} ({dis_id_row})"):
                            if st.button(
                                "Generate explanation",
                                key=f"explain_tab3_{explain_key}",
                            ):
                                try:
                                    with st.spinner("Generating explanation..."):
                                        node_names = build_node_names(data_dict)
                                        graph = load_graph_for_explain(
                                            processed_dir=processed_dir,
                                        )
                                        explanation = predictor.explain_prediction(
                                            dis_id_row, chemical_id_2,
                                            data=graph,
                                            node_names=node_names,
                                        )
                                    st.session_state.tab3_explanations[explain_key] = explanation
                                except Exception as ex:
                                    st.error(f"Explanation error: {ex}")

                            if explain_key in st.session_state.get("tab3_explanations", {}):
                                _render_explanation(st.session_state.tab3_explanations[explain_key])
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Usage:**
        - **Tab 1:** Enter a disease ID and chemical ID to predict their association probability
        - **Tab 2:** Select a disease to find the most likely associated chemicals
        - **Tab 3:** Select a chemical to find the most likely associated diseases
        
        **IDs:**
        - Disease IDs: MESH:DXXXXXX or OMIM:XXXXXX format
        - Chemical IDs: DXXXXXX format (MESH IDs)
        """
    )


if __name__ == "__main__":
    main()
