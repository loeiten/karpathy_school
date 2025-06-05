"""Module containing the verbose_backprop."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable=too-many-locals,too-many-statements,too-many-lines
def succinct_manual_backprop(
    model: Tuple[torch.Tensor, ...],
    intermediate_variables: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    input_data: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Do the manual back propagation, and set the gradients to the parameters.

    Args:
        model (Tuple[torch.Tensor,...]): The weights of the model
        intermediate_variables (Dict[str, torch.Tensor]): The intermediate
            variables (i.e. those which are not part of model parameters).
        targets(torch.Tensor): The targets
            Needed to compute the log_prob gradients
        input_data (torch.Size): The input data for the batch

    Returns:
        A map of the gradients
    """
   # Alias for the model weights
    (
        c,
        w1,
        _,  # We are not using b1 in any calculations
        w2,
        _,  # We are not using b2 in any calculations
        batch_normalization_gain,
        _,  # We are not using batch_normalization_bias
    ) = model
    # Intermediate variables from predict
    embedding = intermediate_variables["embedding"]
    concatenated_embedding = intermediate_variables["concatenated_embedding"]

    inv_batch_normalization_std = intermediate_variables["inv_batch_normalization_std"]
    batch_normalization_raw = intermediate_variables["batch_normalization_raw"]
    h_pre_activation = intermediate_variables["h_pre_activation"]
    h = intermediate_variables["h"]
    # Intermediate variables from loss
    logits = intermediate_variables["logits"]

    batch_size = embedding.size(dim=0)

    # Calculate the gradients
    # Calculate the derivatives of the cross entropy
    #
    # We start from
    # l
    # = - \frac{1}{N} \sum_{n=0}^{N} \sum_{c=0}^{C}
    #   y_{nc}
    #   \log(
    #     \frac{
    #      \exp(x_{nc} - \max_{C}(x_{nc}) )
    #       }{
    #      \sum_{C} \exp(x_{nc} - \max_{C}(x_{nc}))
    #      }
    #   )
    #
    # We now want to understand how l changes when we change x_{nc}.
    # To simplify the notation, we use the same variables as before
    #
    # m_{n}(x_{n1}, \ldots, x_{nC}) = \max_{C} x_{nc}
    # o_{nc}(x_{nc}, m_{n}) = x_{nc} - m_{n}
    # e_{nc}(o_{nc}) = \exp(o_{nc})
    # s_{n}(e_{n1}, \ldots, e_{nC}) = \sum_{j=0}^C e_{nj}
    # i_{n}(s_{n}) = \frac{1}{s_{n}}
    # u_{nc}(e_{nc},i_{n}) = \ln(e_{nc},i_{n}).
    # p_{nc} = \frac{e_{nc}}{s_{n}}
    #
    # so
    # l
    # = -\frac{1}{N} \sum_{n=0}^N \sum_{c=0}^C
    #    y_{nc}
    #    u_{nc}(,
    #      \frac{e_{nc}(o_{nc}(x_{nc},m_{n}))}
    #           {s_{n}(e_{n1},\dots,e_{nC})}
    #    )
    # = -\frac{1}{N} \sum_{n=0}^N \sum_{c=0}^C
    #    y_{nc}
    #    u_{nc}(
    #      e_{nc}(o_{nc}(x_{nc},m_{n})),
    #      i_{n}(s_{n}(e_{n1},\dots,e_{nC}))
    #    )
    #
    # By using the chain rule, we get
    #
    # dl
    # = -\frac{1}{N}\sum_{n=0}^N\sum_{c=0}^C y_{nc}d[u_{nc}(e_{nc},i_{n})]
    # = -\frac{1}{N}\sum_{n,c}y_{nc}
    #    (
    #      \frac{\partial u_{nc}}{\partial e_{nc}} d e_{nc}
    #      +
    #      \frac{\partial u_{nc}}{\partial i_{n}} d i_{n}
    #    )
    #
    # The differentials can now be written as
    #
    # de_{nc}
    # = \frac{\partial e_{nc}}{\partial o_{nc}} d o_{nc}
    # = e_{nc} d o_{nc}
    #
    # from the definition of the derivative of the exponential.
    #
    # d o_{nc}
    # = \frac{\partial o_{nc}}{\partial x_{nc}} d x_{nc}
    #   + \frac{\partial o_{nc}}{\partial m_{n}}d m_{n}
    # = d x_{nc} - d m_{n}
    #
    # from the definition of o_{nc}.
    #
    # d m_{n} = \sum_{k=0}^C \frac{\partial m_{n}}{\partial x_{nk}} dx_{nk}
    #         = \sum_{k=0}^C \{c=\arg \max_{C}x_{nc} \} dx_{nk}
    #
    # from the definition of the \max function.
    #
    # d i_{n}
    # = \frac{\partial i_{n}}{\partial s_{n}} d s_{n} = -\frac{1}{s_{n}^2}d s_{n}
    #
    # from the definition of the derivative to the negative power.
    #
    # d s_{n} = \sum_{j=0}^C \frac{\partial s_{n}}{\partial e_{nj}} d e_{nj}
    #         = \sum_{j=0}^C d e_{nj}
    #
    # as only one of the terms of the sum is going to be non-zero, using the
    # above yields
    #
    # d s_{n} = \sum_{j=0}^C e_{nj}(d x_{nj}-d m_{n})
    #
    # By calculating the partial derivatives, we get
    #
    # \frac{\partial u_{nc}}{\partial e_{nc}} = \frac{1}{e_{nc}}
    # \frac{\partial u_{nc}}{\partial i_{n}} = \frac{1}{i_{n}}
    # \frac{\partial e_{nc}}{\partial o_{nc}} = e_{nc}
    # \frac{\partial i_{n}}{\partial s_{n}} = \frac{1}{s_{n}^2}
    #
    # If we combine what we now have, we get
    #
    # dl
    # = -\frac{1}{N}\sum_{n,c}y_{nc}
    #    [
    #      \frac{\partial u_{nc}}{\partial e_{nc}}
    #      e_{nc}(d x_{nc}-d m_{n})
    #      +
    #      \frac{\partial u_{nc}}{\partial i_{n}}
    #      (-\frac{1}{s_{n}^2}\sum_{j}e_{nj}(d x_{nj}-d m_{n}))
    #    ]
    # = -\frac{1}{N}\sum_{n,c}y_{nc}
    #    [
    #      \frac{1}{e_{nc}} e_{nc}(d x_{nc}-d m_{n})
    #      +
    #      i_{n}
    #      (-\frac{1}{s_{n}^2}\sum_{j}e_{nj}(d x_{nj}-d m_{n}))
    #    ]
    #
    # Using that e_{nc}/e_{nc} = 1 and that i_{n} = 1/s_{n} gives
    #
    # dl
    # = -\frac{1}{N}\sum_{n,c}y_{nc}
    #    [
    #      d x_{nc}-d m_{n}
    #      -
    #      \frac{1}{s_{n}}\sum_{j}e_{nj}(d x_{nj}-d m_{n})
    #    ]
    #
    # Using that e_{nj}/s_{n} = p_{nj} and that sum_j p_{nj} = 1 gives
    #
    # dl
    # = -\frac{1}{N}\sum_{n,c}y_{nc}
    #    [
    #      d x_{nc}-d m_{n}
    #      -
    #      \sum_{j}p_{nj}d x_{nj}-d m_{n}
    #    ]
    # = -\frac{1}{N}\sum_{n,c}y_{nc}
    #    [
    #      d x_{nc}
    #      -
    #     \sum_{j} p_{nj}d x_{nj}
    #    ]
    # = -\frac{1}{N}\sum_{n,j}
    #    [
    #      y_{nj}
    #      -
    #     \sum_{c}y_{nc} p_{nj}
    #    ]d x_{nj}
    #
    # where we have re-indexed the second sum.
    # Using that \sum_{c}y_{nc} = 1 gives
    #
    # dl
    # = -\frac{1}{N}\sum_{n,j} [y_{nc} - p_{nj} ]d x_{nj}
    #
    # So
    #
    # \frac{d l}{d x_{nj}}
    # = -\frac{1}{N}[y_{nj}-p_{nj}]
    # = \frac{1}{N}[p_{nj}-y_{nj}]
    dl_d_logits = F.softmax(logits, 1)  # The p_{nj} part
    dl_d_logits[range(batch_size), targets] -= 1  # -y_{nj} for the correct idxs
    dl_d_logits /= batch_size  # 1/N

    # Reusing the derivation we did in verbose_manual_backprop
    dl_d_h = dl_d_logits @ w2.T
    dl_d_w2 = h.T @ dl_d_logits
    dl_d_b2 = dl_d_logits.sum(0)
    dl_d_h_pre_activation = dl_d_h * (1.0 - torch.tanh(h_pre_activation) ** 2)

