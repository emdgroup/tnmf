import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.demo import SignalTool, st_define_nmf_params


@st.cache(hash_funcs={DeltaGenerator: lambda _: None})
def fit_nmf_model(V, nmf_params, progress_bar):
    fit_params = {k: nmf_params.pop(k) for k in ('sparsity_H', 'inhibition_strength')}
    nmf = TransformInvariantNMF(**nmf_params)
    nmf.fit(V, progress_callback=lambda _, x: progress_bar.progress((x + 1) / nmf_params['n_iterations']), **fit_params)
    return nmf


def main(progress_bar, n_dims: int, verbose: bool = True):
    """
    Runs the streamlit demo on synthetic signals for the specified signal type.

    Parameters
    ----------
    progress_bar
        Streamlit progress bar that needs to be updated during model fitting.
    n_dims : int
        The number of dimensions of the input signals.
    verbose : bool
        If True, show detailed information.
    """
    # show demo info text
    if verbose:
        st.markdown('''
        ## Demo guide
        This dashboard demonstrates the use of the *Transform-Invariant Non-Negative Matrix Factorization (TNMF) package*
        in the specific context of learning **shift-invariant representations**.
        Via the **drop-down menu** on the left, you can choose between different demo examples.

        ## Usage
        Detailed explanations of the available options and the shown output (including this text) can be displayed or hidden
        using the **'Verbose' checkbox**. When the checkbox is unticked, the description of the control widgets can still
        be accessed via the **tooltips** next to them.
        ''')

    # create the signal handling tool
    tool = SignalTool(n_dims=n_dims)

    # generate random input data for the factorization
    V, opt_nmf_params = tool.st_generate_input(verbose=verbose)

    # define the NMF parameters and fit the model
    nmf_params = st_define_nmf_params(opt_nmf_params, verbose=verbose)
    nmf = fit_nmf_model(V, nmf_params, progress_bar)

    # visualize the results
    tool.st_compare_signals(V, nmf.R, verbose=verbose)
    tool.st_compare_individual_signals(V, nmf.R, verbose=verbose)
    tool.st_plot_partial_reconstructions(V, nmf, verbose=verbose)
