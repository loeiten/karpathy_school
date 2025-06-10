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

    # We can also calculate the batch norm all in one go
    # Using the definitions as before
    #
    # \mu_{h} = \mu_{h}(d_{1h}, \dots ,d_{Nh})
    #         = \frac{1}{N}\sum_{i=0}^N d_{ih}
    # f_{nh} = f(d_{nh},\mu_{h})
    #        = d_{nh} - \mu_{h}
    # \sigma_{h} = s_h(f_{1h}, \dots, f_{Nh})
    #           = \frac{1}{N-1}\sum_{i=0}^N f_{ih}^2
    # j_{h} = j(\sigma_{h})
    #       = \frac{1}{\sqrt{\sigma_{h} + \epsilon}}
    # k_{nh} = k(f_{nh},j_{h})
    #        = f_{nh}j_{h}
    # a_{nh} = a(\gamma_{h}, k_{nh}, \beta_{h})
    #        = \gamma_{h},k_{nh} + \beta_{h}
    #
    # Or, stated differently:
    #
    # a_{nh} =
    # a_{nh}(\gamma_{h}, \beta_{h}, k_{nh}(f_{nh}(d_{nh}, \mu_{h}), j_{h}(\sigma_{h})))
    #
    # Expanding the differential gives
    #
    # dl
    # = \sum_{i=0}^N \frac{\partial l}{\partial a_{ih}} d a_{ih}
    #
    # where
    #
    # d a_{ih}
    # = \frac{\partial a_{ih}}{\partial \gamma_{h}} d\gamma_{h}
    #  + \frac{\partial a_{ih}}{\partial k_{ih}} d k_{ih}
    #  + \frac{\partial a_{ih}}{\partial \beta_{h}} d\beta_{h}
    # = \gamma_{h} d k_{ih}
    #
    # where we in the last step have used that \gamma and \beta are constants
    # w.r.t d_{nh}, so d\gamma_{h} = d\beta_{h} = 0.
    # Furthermore, we have that
    #
    # d k_{ih}
    # = \frac{\partial k_{ih}}{\partial f_{ih}} d f_{ih}
    #  + \frac{\partial k_{ih}}{\partial j_{h}} d j_{h}
    # = j_{h} d f_{ih} + f_{ih} d j_{h}
    #
    # and
    #
    # d f_{ih}
    # = d d_{ih} - d \mu_{h}
    # = d d_{ih} - \frac{1}{N}\sum_{p=0}^N d d_{ph}
    #
    # and
    #
    # d j_{h}
    # = \frac{\partial j_{h}}{\partial \sigma_{h}} d \sigma_{h}
    # = -\frac{1}{2}(\sigma_{h}+\epsilon)^{-\frac{3}{2}} d\sigma_{h}
    # = -\frac{1}{2} j_{h}^3 d\sigma_{h}
    #
    # and
    #
    # d\sigma_{h}
    # = \frac{\partial \sigma_{h}}{\partial f_{jh}} d f_{jh}
    # = \frac{1}{N-1}\sum_{j=0}^N 2 f_{jh} d f_{jh}
    # = \frac{2}{N-1}\sum_{j=0}^N f_{jh} (d d_{jh}-d\mu_{h})
    #
    # Combining all the differentials gives
    #
    # dl
    # = \sum_{i=0}^N \frac{\partial l}{\partial a_{ih}}\gamma_{h}
    #    [
    #      j_{h}(d d_{ih}-d\mu_{h})
    #      -
    #      \frac{j_{h}^3 f_{ih}}{2} \frac{2}{N-1}
    #      \sum_{j=0}^N f_{jh}(d d_{jh}-d\mu_{h})
    #    ]
    # = \gamma_{h}
    #   \sum_{i=0}^N \frac{\partial l}{\partial a_{ih}}
    #   [
    #     j_{h}(d d_{ih}-d\mu_{h})
    #     -
    #     \frac{j_{h}^3 f_{ih}}{N-1}
    #     \sum_{j=0}^N f_{jh}(d d_{jh}-d\mu_{h})
    #   ]
    # = \gamma_{h}
    #   \sum_{i=0}^N \frac{\partial l}{\partial a_{ih}}
    #   [
    #     j_{h}(d d_{ih}-\frac{1}{N}\sum_p d d_{ph})
    #     -
    #     \frac{j_{h}^3 f_{ih}}{N-1}
    #     \sum_{j=0}^N f_{jh}(d d_{jh}-\frac{1}{N}\sum_p d d_{ph})
    #   ]
    # = \sum_{i=0}^N
    #   [
    #     \gamma_{h}j_{h}\frac{\partial l}{\partial a_{ih}}
    #     d d_{ih}
    #     -
    #     \gamma_{h}j_{h}\frac{\partial l}{\partial a_{ih}}
    #     \frac{1}{N}\sum_p
    #     d d_{ph}
    #     -
    #     \gamma_{h}\frac{\partial l}{\partial a_{ih}}
    #     \frac{j_{h}^3 f_{ih}}{N-1}
    #     \sum_{j=0}^N f_{jh}d d_{jh}
    #     -
    #     \gamma_{h}\frac{\partial l}{\partial a_{ih}}
    #     \frac{j_{h}^3 f_{ih}}{N-1}
    #     \sum_{j=0}^N f_{jh}\frac{1}{N}\sum_p d d_{ph}
    #   ]
    #
    # We note that
    #
    # \sum_j f_jh
    # = \sum_j (d_{jh} - \mu_{h})
    # = \sum_j d_{jh} - \sum_j\mu_{h}
    # = N\mu_{h} - N\mu_{h}
    # = 0
    #
    # At the same time, \sum_{j=0}^N f_{jh}d d_{jh} \neq 0, so
    #
    # dl
    # = \gamma_{h}j_{h}
    #   \sum_{i=0}^N
    #   \frac{\partial l}{\partial a_{ih}}
    #   d d_{ih}
    #     -
    #   \gamma_{h}j_{h}
    #   \sum_{i=0}^N
    #   \frac{1}{N}
    #   \frac{\partial l}{\partial a_{ih}}
    #   \sum_p
    #   d d_{ph}
    #     -
    #   \gamma_{h}\frac{j_{h}^3 }{N-1}
    #   \sum_{i=0}^N
    #   \frac{\partial l}{\partial a_{ih}}
    #   f_{ih}
    #   \sum_{j=0}^N f_{jh}d d_{jh}
    # = \sum_{n=0}^N
    #   (
    #   \gamma_{h}j_{h}
    #   \frac{\partial l}{\partial a_{nh}}
    #   )
    #   d d_{nh}
    #     -
    #   \frac{1}{N}
    #   \gamma_{h}j_{h}
    #   \sum_p
    #   (
    #   \sum_{i=0}^N
    #   \frac{\partial l}{\partial a_{ih}}
    #   )
    #   d d_{ph}
    #     -
    #   \gamma_{h}\frac{j_{h}^3 }{N-1}
    #   \sum_{j=0}^N
    #   (
    #   \sum_{i=0}^N
    #   \frac{\partial l}{\partial a_{ih}} f_{ih}
    #   )
    #   f_{jh}d d_{jh}
    # = \sum_{n=0}^N
    #   (
    #   \gamma_{h}j_{h}
    #   \frac{\partial l}{\partial a_{nh}}
    #   )
    #   d d_{nh}
    #     -
    #   \sum_{n=0}^N
    #   (
    #   \gamma_{h}j_{h}
    #   \frac{1}{N}
    #   \sum_{i=0}^N
    #   \frac{\partial l}{\partial a_{ih}}
    #   )
    #   d d_{nh}
    #     -
    #   \sum_{n=0}^N
    #   \gamma_{h}\frac{j_{h}^3 }{N-1}
    #   f_{nh}
    #   (
    #   \sum_{i=0}^N
    #   \frac{\partial l}{\partial a_{ih}} f_{ih}
    #   )
    #   d d_{nh}
    # = \sum_{n=0}^N
    #   [
    #     \gamma_{h}j_{h}
    #     \frac{\partial l}{\partial a_{nh}}
    #       -
    #     \gamma_{h}j_{h}
    #     \frac{1}{N}
    #     \sum_{i=0}^N
    #     \frac{\partial l}{\partial a_{ih}}
    #       -
    #     \gamma_{h}\frac{j_{h}^3 }{N-1}
    #     f_{nh}
    #     \sum_{i=0}^N
    #     f_{ih}
    #     \frac{\partial l}{\partial a_{ih}}
    #   ]
    #   d d_{nh}
    #
    # Reading off the coefficient yields
    #
    # \frac{dl}{d d_{nh}}
    # = \gamma_{h}j_{h}
    #   [
    #     \frac{\partial l}{\partial a_{nh}}
    #       -
    #     \frac{1}{N}
    #     \sum_{i=0}^N
    #     \frac{\partial l}{\partial a_{ih}}
    #       -
    #     \frac{j_{h}^2 }{N-1}
    #     f_{nh}
    #     \sum_{i=0}^N
    #     f_{ih}
    #     \frac{\partial l}{\partial a_{ih}}
    #   ]
    # = \frac{\gamma_{h}j_{h}}{N}
    #   [
    #     N\frac{\partial l}{\partial a_{nh}}
    #       -
    #     \sum_{i=0}^N
    #     \frac{\partial l}{\partial a_{ih}}
    #       -
    #     \frac{N}{N-1}
    #     j_{h}^2
    #     f_{nh}
    #     \sum_{i=0}^N
    #     f_{ih}
    #     \frac{\partial l}{\partial a_{ih}}
    #   ]
    #
    # Using that k_{nh} = f_{nh}j_{h} gives
    #
    # \frac{dl}{d d_{nh}}
    # = \frac{\gamma_{h}j_{h}}{N}
    #   [
    #     N\frac{\partial l}{\partial a_{nh}}
    #       -
    #     \sum_{i=0}^N
    #     \frac{\partial l}{\partial a_{ih}}
    #       -
    #     \frac{N}{N-1}
    #     k_{nh}
    #     \sum_{i=0}^N
    #     k_{ih}
    #     \frac{\partial l}{\partial a_{ih}}
    #   ]
    dl_d_h_pre_batch_norm = (
        batch_normalization_gain
        * inv_batch_normalization_std
        / batch_size
        * (
            batch_size * dl_d_h_pre_activation
            - dl_d_h_pre_activation.sum(0)
            - batch_size
            / (batch_size - 1)
            * batch_normalization_raw
            * (dl_d_h_pre_activation * batch_normalization_raw).sum(0)
        )
    )

    # Reusing the derivation we did in verbose_manual_backprop
    dl_d_batch_normalization_gain = (
        dl_d_h_pre_activation * batch_normalization_raw
    ).sum(0, keepdim=True)
    dl_d_batch_normalization_bias = (dl_d_h_pre_activation).sum(0, keepdim=True)

    # And for the first layer, we have
    dl_d_w1 = concatenated_embedding.T @ dl_d_h_pre_batch_norm
    dl_d_b1 = dl_d_h_pre_batch_norm.sum(0)
    dl_d_concatenated_embedding = dl_d_h_pre_batch_norm @ w1.T
    dl_d_embedding = dl_d_concatenated_embedding.view(
        batch_size, -1, embedding.shape[2]
    )
    dl_d_c = torch.zeros_like(c)
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            idx = input_data[i, j]
            dl_d_c[idx] += dl_d_embedding[i, j]

    gradients: Dict[str, torch.Tensor] = {}
    gradients["dl_d_logits"] = dl_d_logits
    gradients["dl_d_h"] = dl_d_h
    gradients["dl_d_h_pre_activation"] = dl_d_h_pre_activation
    gradients["dl_d_w2"] = dl_d_w2
    gradients["dl_d_b2"] = dl_d_b2
    gradients["dl_d_h_pre_batch_norm"] = dl_d_h_pre_batch_norm
    gradients["dl_d_batch_normalization_gain"] = dl_d_batch_normalization_gain
    gradients["dl_d_batch_normalization_bias"] = dl_d_batch_normalization_bias
    gradients["dl_d_w1"] = dl_d_w1
    gradients["dl_d_b1"] = dl_d_b1
    gradients["dl_d_concatenated_embedding"] = dl_d_concatenated_embedding
    gradients["dl_d_embedding"] = dl_d_embedding
    gradients["dl_d_c"] = dl_d_c

