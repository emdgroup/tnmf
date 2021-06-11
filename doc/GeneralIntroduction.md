# General Introduction

The purpose of NMF is to learn *parts-based representations* of data, which is achieved by separating the data into a set of **dictionary elements** and corresponding **activations** (see [1]_).
Both the dictionary elements and their activations are required to be *non-negative*, such that the induced superposition of dictionary elements (weighted with their corresponding activation terms) reconstructs the data in a purely *additive* way.
This has the effect that characteristic features emerging in the dictionary during the learning process must correspond to meaningful parts of the data, since each individual feature can only be added to the reconstruction the data but not be subtracted from it.

## Notation
TODO: add section

## Non-Negative Matrix Factorization

In its simplest variant, the NMF task can be formulated as a pure matrix factorization problem, where the data is represented by a non-negative matrix :math:`V \in \mathbb{R}_{\geq 0}^{S \times D}` that is to be approximated through a product of a non-negative dictionary matrix :math:`W \in \mathbb{R}_{\geq 0}^{K \times D}` and a non-negative activation matrix :math:`H \in \mathbb{R}_{\geq 0}^{S \times K}`,

.. math::
    V \approx H W.  
    :label: nmf

.. note::
    In contrast to most of the NMF literature, we represent individual data points by row vectors instead of column vectors in order to be consistent with the row-major data representation used in the code.

By defining an appropriate divergence measure :math:`D`, the factorization task can be translated into a proper optimization problem of the following form (see [2]_),

.. math::
    \min_{W, H} D(V \mid H W) \quad \text{subject to} \quad W \geq 0, H \geq 0.

A common choice is the *Frobenius norm*, which measures the quadratic difference between the data and its reconstruction,

.. math::
    D(V \mid R) = \lVert V - R \rVert_F = \sqrt{ \sum_{s=1}^S \sum_{d=1}^D \lvert V_{sd} - R_{sd} \rvert^2 }.


## Sparse Coding
TODO: add section

## Transform Invariance

Abstractly speaking, the dictionary matrix :math:`W` in Equation :eq:`nmf` contains :math:`K` *characteristic features* represented through its row vectors :math:`\lbrace W_k \rbrace`, which are superimposed via the corresponding activation vector :math:`H_s` to form the input sample :math:`V_s`,

.. math::
    V_{s} \approx \sum_k H_{sk} W_{k}.
    :label: nmf_synthesis


As can be seen from the above equation, the individual dictionary elements :math:`\lbrace W_k \rbrace` have the same size as the samples :math:`\lbrace V_s \rbrace`.
In many applications, however, typical features contained in the data are smaller than the individual samples and exhibit certain kinds of *transform invariance*.

For example, image data is typically composed of smaller constituents, which represent different parts of objects and can contribute to the image at all possible locations on the pixel grid.
This particular degree of freedom stems from the simple fact that objects can usually move freely within a scene and can hence appear at different locations in the recorded image, rendering the characteristic image features *invariant under change of location* (shift invariance).
Other types of invariances related to image data arise from additional spatial transforms of the involved objects depicted in the scene (such as scaling, rotation, mirroring) or changes in the lightning conditions and the measurement process (e.g. change of color or contrast).
In general, invariances can be also observed in other types of data, such as audio recordings, where each individual characteristic feature (e.g. a tone) belongs to a larger part (a chord), which in turn may occur in different timbres, in different keys, for different durations, and so on.

Instead of attempting to capture all possible instantiations of the involved dictionary elements that could be generated through their applicable transforms (which would require an exponentially large dictionary), a more data-efficient approach is to decouple the transforms from their dictionary elements and learn a *transform-invariant dictionary*.
This can be achieved by encoding the transforms explicitly into the model,

.. math::
    V_{s} = \sum_k \sum_m H_{smk} T_m[\tilde{W}_{k}].
    :label: tnmf_synthesis

Herein, the set of possible transforms of a given dictionary element :math:`\tilde{W}_k` (which in the following is referred to as an **elementary atom**) is described through a **transform operator** :math:`T : \mathbb{R}_{\geq 0}^L \times \lbrace 1, \ldots, M \rbrace \rightarrow \mathbb{R}_{\geq 0}^S`, which can be indexed to refer to a particular instantiation of the transform.
In the image case, for instance, :math:`T` could describe all possible shifts of a smaller image patch within an image region, with :math:`T_m` corresponding to a particular shift of the patch to a specific location on the pixel grid.
The corresponding activations are stored in an **activation tensor** :math:`H \in \mathbb{R}_{\geq 0}^{S \times M \times K}`, whose element :math:`H_{smk}` quantifies the contribution of the :math:`m`-th transform of the :math:`k`-th dictionary element to the :math:`s`-th data sample.
Note, in particular, that the sizes of :math:`V_s` and :math:`W_k` are no longer coupled in this model since the transform operator :math:`T` maps each dictionary element from a separate **latent space** :math:`\mathbb{R}_{\geq 0}^L`, whose dimensionality :math:`L` can be defined independently, to the sample space :math:`\mathbb{R}_{\geq 0}^D`.

For the data reconstruction part, the synthesis procedure in Equation :eq:`tnmf_synthesis` is, in fact, equivalent to that of Equation :eq:`nmf_synthesis` when using an extended dictionary :math:`W` that contains all possible transforms of the original elements.
However, the important difference to note is that each dictionary element of that extended dictionary would be considered an independent parameter in the latter approach whereas all transformed versions of the elements are coupled through their elementary atoms :math:`\lbrace \tilde{W}_k \rbrace` and hence need to be identified through the same shared parameters.

TODO: add documentation on inhibition regularization term

## Multi-channel Data
TODO: add section

## References

.. [1] Lee, D.D., Seung, H.S., 2000. Algorithms for Non-negative Matrix Factorization,
    in: Proceedings of the 13th International Conference on Neural Information
    Processing Systems. pp. 535–541. https://doi.org/10.5555/3008751.3008829

.. [2] Févotte, C., & Idier, J, 2011. Algorithms for Nonnegative Matrix Factorization with the β-divergence.
    Neural computation, 23(9), pp. 2421-2456. https://doi.org/10.1162/NECO_a_00168

## Purpose of this Package
This package provides a toolset to learn invariant data representations of the form described in Equation :eq:`tnmf_synthesis` for arbitrary transform types, i.e., it can be used to find the latent dictionary "behind" the underlying transform.

In the specific case of image data and shift invariance (to mention only one of many possible combinations of data and transforms), the package allows to extract a dictionary of image patches that reconstruct a given image through a specific arrangement of their shifted versions.
In this sense, it allows to "undo" the data-generating transform operation so that the learned dictionary encodes the input *modulo* shift.
