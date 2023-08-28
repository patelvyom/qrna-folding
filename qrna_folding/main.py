import sys
from typing import List

import experiments as experiments
import preprocessors
import streamlit as st
import visualisation as viz


def is_rna_valid(sequence: str) -> bool:
    valid_chars = set("AUGC")
    return all(base in valid_chars for base in sequence)


def main(argv: List[str]) -> None:
    st.title("Quantum RNA Folding")
    rna_sequence = st.text_area(
        "Enter RNA Sequence",
        "CGGUUUUCCUAAAUCGUCGUGAGACGGACCUUGGAAAACCUAGUAUUUAAAUAACUAGGUC",
        max_chars=128,
    )
    if rna_sequence:
        st.markdown("#### Entered Sequence:")
        st.code(rna_sequence, language="text")
        assert is_rna_valid(rna_sequence), f"RNA sequence {rna_sequence} is invalid."

    preprocessor = preprocessors.NormalStemLengthPreProcessor(rna_sequence)
    min_stem_length = int(st.text_input("Enter minimum stem length", "5"))

    if st.button("Perform Preprocessing"):
        preprocessor.process(min_stem_length=min_stem_length)

        st.markdown("#### Adjacency Matrix")
        ax = viz.plot_adjacency_matrix(
            preprocessor.adjacency_matrix, rna_seq=rna_sequence
        )
        st.pyplot(ax.figure)

        st.markdown("### Potential Stems")
        st.markdown(f"#### Found {preprocessor.n_potential_stems} potential stems.")

    st.markdown("### Selected Stems")
    n_stems_to_select = int(st.text_input("Enter number of stems to select", "20"))
    selection_method = st.selectbox("Choose selection method", ["longest", "random"])
    preprocessor.select_n_stems(n_stems_to_select, method=selection_method)

    if st.button("Visualise Selected Stems"):
        for stem in preprocessor.selected_stems:
            ax = viz.generate_network_graph_image(rna_sequence, stem)
            st.pyplot(ax.figure)

    n_layers = int(st.text_input("Enter number of layers", "5"))
    device = st.selectbox("Choose device", ["default.qubit", "lightning.qubit"])
    if st.button("Run Experiment"):
        experiment = experiments.HamiltonianV1(
            preprocessor, circuit_depth=n_layers, eps=6, c_p=0, device=device
        )
        experiment.run()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
