from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.demo import SignalTool, st_define_nmf_params


def main(progress_bar, n_dims: int):
    """
    Runs the streamlit demo on synthetic signals for the specified signal type.

    Parameters
    ----------
    progress_bar
        Streamlit progress bar that needs to be updated during model fitting.
    n_dims : int
        The number of dimensions of the input signals.
    """
    # create the signal handling tool
    tool = SignalTool(n_dims=n_dims)

    # generate random input data for the factorization
    V = tool.st_generate_input()

    # define the NMF parameters and fit the model
    nmf_params = st_define_nmf_params(n_dims=n_dims)
    nmf = TransformInvariantNMF(**nmf_params)
    nmf.fit(V, progress_callback= lambda _, x: progress_bar.progress((x + 1) / nmf_params['n_iterations']))

    # visualize the results
    tool.st_compare_signals(V, nmf.R)
    tool.st_compare_individual_signals(V, nmf.R)