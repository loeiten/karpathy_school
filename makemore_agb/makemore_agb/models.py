"""Module for models."""

from typing import List, Optional, Tuple

import torch

from makemore_agb import DEVICE, VOCAB_SIZE


# Reducing the number of locals here will penalize the didactical purpose
# pylint: disable-next=too-many-locals,too-many-arguments
def get_explicit_model(
    block_size: int,
    embedding_size: int = 2,
    hidden_layer_neurons: int = 100,
    seed: int = 2147483647,
    good_initialization: bool = True,
    batch_normalize: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Return the explicit model.

    Args:
        block_size (int): Number of input features to the network
            This is how many characters we are considering simultaneously, aka.
            the context length
        embedding_size (int): The size of the embedding
        hidden_layer_neurons (int): The seed for the random number generator
        seed (int): The seed for the random number generator
        good_initialization (bool): Whether or not to use an initialization
            which has a good distribution of the initial weights
        batch_normalize (bool): Whether or not to include batch normalization
            parameters

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the parameters of the
            neural net.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    # NOTE: randn draws from normal distribution, whereas rand draws from a
    #       uniform distribution
    c = torch.randn(
        (VOCAB_SIZE, embedding_size), generator=g, requires_grad=True, device=DEVICE
    )
    w1 = torch.randn(
        (block_size * embedding_size, hidden_layer_neurons),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    b1 = torch.randn(
        hidden_layer_neurons, generator=g, requires_grad=True, device=DEVICE
    )
    w2 = torch.randn(
        (hidden_layer_neurons, VOCAB_SIZE),
        generator=g,
        requires_grad=True,
        device=DEVICE,
    )
    b2 = torch.randn(VOCAB_SIZE, generator=g, requires_grad=True, device=DEVICE)

    if good_initialization:
        # Initially the model is confidently wrong, that is: The probability
        # distribution of the output is not uniform
        # Recall that the logits are given as h @ w2 + b2
        # Taking the softmax of the logits gives the probability
        # The negative of the logarithm of the softmax gives the loss
        # If the model has a high value for the logit of the correct character
        # compared to all the others this will give a low loss
        # If on the other hand it assigns a high probability to the wrong
        # character, this will result in a high loss
        # If we have initialized with uniform probability, the probability of
        # drawing a specific character would be 1/VOCAB_SIZE
        # If vocab size is 27, this means that -log(1/27)=3.2958 is the log that
        # we would expect.
        # However, we usually observe a much higher loss initially as the
        # initialization by chance are favouring some characters
        # Initially we would like the logits to be close to zero due to
        # numerical stability
        b2.data *= 0
        # As we want the logits to be small, we could have set the initial
        # weights to 0
        # However, we want to break the symmetry for better training, so it's
        # better to have some entropy with numbers close to 0
        # Since we want the weights to be well behaved in both the forward and
        # the backward propagation, we set the initialization to the Kaiming
        # initialization
        w2.data *= (5 / 3) / (hidden_layer_neurons**0.5)
        # We could have some small entropy in the weights as well
        b1.data *= 0.01
        # In the pre-activation we are multiplying the embedding with some
        # random weights w1
        # This causes the product to broaden the distribution
        # Since the distribution is broad, a lot of values are in the extremes
        # of tanh
        # Looking at the gradient, we see that
        # grad = (1 - tanh(h)**2) * out.grad
        # If tanh(h) becomes -1 or 1, then the gradient becomes 0, killing all
        # possibilities to learn
        # Hence, we need to squash the distribution so that we don't hit the
        # extremes of tanh
        # It would be nice to have the standard deviation of the pre-activation
        # would be around 1 so that tanh in the activation doesn't take on
        # extreme values
        # Kaiming initialization scales the distribution so that the standard
        # deviation are well behaved both in the forward and the backward
        # propagation
        w1.data *= (5 / 3) / ((block_size * embedding_size) ** 0.5)

    parameters = [c, w1, b1, w2, b2]

    if batch_normalize:
        # We would like to normalize each batch after each layer so that it's
        # roughly normal
        # However, only having normal distribution would yield poor results
        # Hence we let the gain and bias be trainable parameters the network can use
        # in order to move the distribution around
        batch_normalization_gain = torch.ones((1, hidden_layer_neurons), device=DEVICE)
        batch_normalization_bias = torch.zeros((1, hidden_layer_neurons), device=DEVICE)

        parameters.append(batch_normalization_gain)
        parameters.append(batch_normalization_bias)

    # Make it possible to train
    for p in parameters:
        p.requires_grad = True

    print(
        f"Number of elements in model: {sum(layer.nelement() for layer in parameters)}"
    )

    return tuple(parameters)


class Linear:
    """Class mimicking the torch.nn.Linear Module in PyTorch."""

    # Reducing the number of arguments here would be counter productive
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        seed: int = 2147483647,
    ):
        """Set the weights and biases for the linear layer.

        Args:
            fan_in (int): Number of inputs to the layer.
            fan_out (int): Number of outputs from the layer.
            bias (bool, optional): Whether or not to use the bias term.
                Defaults to True.
            device (Optional[torch.device], optional): Device to use for the tensors.
                Defaults to None.
            seed (int): The seed to use in the random number generator
        """
        g = torch.Generator(device=device).manual_seed(seed)
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        # NOTE: This is not implemented in the original torch layer, but is added
        #       for plotting convenience
        self.out = torch.empty(fan_out, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return the parameters.

        Returns:
            List[torch.Tensor]: The parameters of the layer
        """
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Tanh:
    """Class mimicking the torch.nn.Tanh Module in PyTorch."""

    # Reducing the number of arguments here would be counter productive
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        """Initialize the output for the tanh layer.

        Args:
            device (Optional[torch.device], optional): Device to use for the tensors.
        """
        # NOTE: This is not implemented in the original torch layer, but is added
        #       for plotting convenience
        self.out = torch.empty(0, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return the parameters.

        Returns:
            List[torch.Tensor]: The parameters of the layer
        """
        return []


# Reducing the number of attributes here will penalize the didactical purpose
# pylint: disable-next=too-many-instance-attributes
class BatchNorm1d:
    """Class mimicking the torch.nn.BatchNorm1d Module in PyTorch."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """Initialize the parameters and the buffers for the normalization.

        Args:
            dim (int): The size of the layer
            eps (float, optional): Epsilon to avoid division by zero. Defaults to 1e-5.
            momentum (float, optional): Momentum to use in the buffer update.
                Defaults to 0.1.
            device (Optional[torch.device], optional): Device to use for the tensors.
        """
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Initialize the parameters
        # The batch normalization gain
        self.gamma = torch.ones(dim, device=device)
        # The batch normalization bias
        self.beta = torch.zeros(dim, device=device)

        # Initialize the buffers
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

        # Initialize the output
        self.out = torch.empty(dim, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the layer given on the input.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        # Forward pass
        if self.training:
            # Mean of the batch
            x_mean = x.mean(0, keepdim=True)
            # Variance of the batch
            x_var = x.var(0, keepdim=True)
        else:
            # Use the buffers
            x_mean = self.running_mean
            x_var = self.running_var

        # Normalize to unit variance
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        # Update the running buffers
        if self.training:
            with torch.no_grad:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * x_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self) -> List[torch.Tensor]:
        """Return the parameters.

        Returns:
            List[torch.Tensor]: The parameters of the layer
        """
        return [self.gamma, self.beta]
