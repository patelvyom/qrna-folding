import sys
from typing import List

import experiments as experiments
import preprocessors
import streamlit as st
import visualisation as viz

key_button_preprocessing = "key_button_preprocessing"
key_button_stem_selection = "key_button_stem_selection"
key_button_experiment = "key_button_experiment"

session_state_vars = {
    "preprocessed": False,
    "stems_selected": False,
    "experiment_finished": False,
    "experiment_results": None,
}


def is_rna_valid():
    rna_sequence = st.session_state.rna_sequence
    valid_chars = set("AUGC")
    assert all(
        base in valid_chars for base in rna_sequence
    ), f"RNA sequence {rna_sequence} is invalid."


def preprocess():
    st.session_state.preprocessed = True
    # Choose preprocessor based on selection
    preprocessor_type = st.session_state.get("preprocessor_type", "Normal (BP count)")
    if preprocessor_type == "H-Bond count":
        st.session_state.preprocessor = preprocessors.HBondCountPreProcessor(
            st.session_state.rna_sequence
        )
    else:
        st.session_state.preprocessor = preprocessors.NormalStemLengthPreProcessor(
            st.session_state.rna_sequence
        )
    st.session_state.preprocessor.process(
        min_stem_length=st.session_state.min_stem_length
    )
    st.session_state.ax = viz.plot_adjacency_matrix(
        st.session_state.preprocessor.adjacency_matrix,
        rna_seq=st.session_state.rna_sequence,
    )


def select_stems():
    st.session_state.stems_selected = True
    n_stems_to_select = st.session_state.n_stems_to_select
    selection_method = st.session_state.selection_method
    preprocessor = st.session_state.preprocessor
    rna_sequence = st.session_state.rna_sequence
    preprocessor.select_n_stems(n_stems_to_select, method=selection_method)
    st.session_state.graphs = []
    for stem in preprocessor.selected_stems:
        ax = viz.generate_network_graph_image(rna_sequence, stem)
        st.session_state.graphs.append(ax)


def run_exp():
    st.session_state.experiment_finished = True
    experiment = experiments.HamiltonianV1(
        st.session_state.preprocessor,
        circuit_depth=st.session_state.n_layers,
        eps=st.session_state.eps,
        c_p=st.session_state.c_p,
        optimizer_steps=st.session_state.optimizer_steps,
        device=st.session_state.device,
    )
    results = experiment.run()
    st.session_state.experiment_results = results


def main(argv: List[str]) -> None:
    st.title("Quantum RNA Folding")

    for var, val in session_state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = val

    rna_sequence = st.text_area(
        "Enter RNA Sequence",
        "CGAAUCUCAAGCAAUCAAGCAUUCUACUUCUAUUGCA",
        max_chars=128,
        key="rna_sequence",
        on_change=is_rna_valid,
    )
    st.markdown("#### Entered Sequence:")
    st.code(rna_sequence, language="text")

    st.sidebar.markdown("# Preprocessing")
    st.sidebar.selectbox(
        "Preprocessor type",
        ["Normal (BP count)", "H-Bond count"],
        key="preprocessor_type",
        help="Normal: stem length = base pair count. H-Bond: stem length = hydrogen bond count.",
    )
    st.sidebar.number_input(
        "Enter minimum stem length", key="min_stem_length", min_value=3
    )
    st.sidebar.button(
        "Perform Preprocessing", key=key_button_preprocessing, on_click=preprocess
    )

    if st.session_state.preprocessed:
        st.markdown("#### Adjacency Matrix")
        st.pyplot(st.session_state.ax.figure)
        st.markdown(
            f"#### Found {st.session_state.preprocessor.n_potential_stems} potential stems."
        )

        st.sidebar.markdown("# Stem Selection")
        st.sidebar.number_input(
            "Enter no. of stems to select",
            key="n_stems_to_select",
            max_value=st.session_state.preprocessor.n_potential_stems,
        )
        st.sidebar.selectbox(
            "Choose selection method", ["longest", "random"], key="selection_method"
        )
        st.sidebar.button(
            "Select Stems", key=key_button_stem_selection, on_click=select_stems
        )

    if st.session_state.stems_selected:
        for plot in st.session_state.graphs:
            st.pyplot(plot.figure)

        st.sidebar.markdown("# QAOA Experiment")
        st.sidebar.number_input(
            "Enter no. of QAOA layers",
            min_value=2,
            max_value=9,
            value=5,
            key="n_layers",
        )
        st.sidebar.selectbox(
            "Choose device", ["default.qubit", "lightning.qubit"], key="device"
        )

        # Advanced parameters
        with st.sidebar.expander("Advanced Parameters"):
            st.number_input(
                "Epsilon (stem normalization)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=1.0,
                key="eps",
                help="Controls stem length normalization in Hamiltonian.",
            )
            st.number_input(
                "Pseudoknot penalty (c_p)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key="c_p",
                help="Penalty coefficient for pseudoknotted stems. 0 = no penalty.",
            )
            st.number_input(
                "Optimizer steps",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="optimizer_steps",
                help="Number of Adam optimization iterations.",
            )

        st.sidebar.button("Run Experiment", key=key_button_experiment, on_click=run_exp)

    if st.session_state.experiment_finished and st.session_state.experiment_results:
        results = st.session_state.experiment_results
        st.markdown("## Results")

        # Convergence plot
        st.markdown("### Optimization Convergence")
        conv_ax = viz.plot_convergence(results["costs"])
        st.pyplot(conv_ax.figure)

        # Best solution
        st.markdown("### Optimal Solution")
        st.markdown(f"**Best bitstring:** `{results['best_bitstring']}`")

        # Interpret bitstring
        all_stems = st.session_state.preprocessor.selected_stems
        selected_stems = results["selected_stems"]
        st.markdown(f"**Selected {len(selected_stems)} stems out of {len(all_stems)}:**")

        if selected_stems:
            stem_table = []
            for i, stem in enumerate(selected_stems):
                stem_table.append({
                    "Stem": i + 1,
                    "Start": stem[0],
                    "End": stem[1],
                    "Length": stem[2],
                })
            st.table(stem_table)

            # Final structure visualization
            st.markdown("### Predicted Secondary Structure")
            struct_ax = viz.generate_final_structure_graph(
                st.session_state.rna_sequence, selected_stems
            )
            st.pyplot(struct_ax.figure)
        else:
            st.info("No stems selected in optimal solution (all bits = 0).")

        # Probability distribution
        st.markdown("### Solution Probability Distribution")
        prob_ax = viz.plot_probabilities(results["probabilities"], top_k=10)
        st.pyplot(prob_ax.figure)

        # Summary stats
        st.markdown("### Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Cost", f"{results['costs'][-1]:.4f}")
        with col2:
            st.metric("Min Cost", f"{min(results['costs']):.4f}")
        with col3:
            prob_best = results["probabilities"].get(results["best_bitstring"], 0)
            st.metric("Best Solution Prob", f"{prob_best:.2%}")

    return


if __name__ == "__main__":
    main(sys.argv[1:])
