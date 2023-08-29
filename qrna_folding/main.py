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
}


def is_rna_valid():
    rna_sequence = st.session_state.rna_sequence
    valid_chars = set("AUGC")
    assert all(
        base in valid_chars for base in rna_sequence
    ), f"RNA sequence {rna_sequence} is invalid."


def preprocess():
    st.session_state.preprocessed = True
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
        eps=6,
        c_p=0,
        device=st.session_state.device,
    )
    experiment.run()


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
        st.sidebar.button("Run Experiment", key=key_button_experiment, on_click=run_exp)

    if st.session_state.experiment_finished:
        st.markdown("## Results")

    return


if __name__ == "__main__":
    main(sys.argv[1:])
