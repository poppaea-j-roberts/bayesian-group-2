"""Implements flow matching posterior estimation."""

from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *

# inputs: samples from joint, samples from some prior, N(0,1), model
# hyperparameters:
# output: samples for each time

class DataConditionalModel(torch.nn.Module):
    """ for an input model, fixes final input values to values specified in data
    in FMPE setting, equivalent to conditioning a model on fixed data following training
    in practice, can be used like a separate model which operates on fewer inputs """
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def forward(self, param):
        return self.model(torch.cat([param, self.data], dim=1))


def fit_FMPE_model(joint_samples, noise_samples, model=None, flow_matching=None,
                   batch_size = 256):
    """
    trains model on conditional flows from noise samples to joint samples
    Args:
        joint_samples: tuple of (parameter samples, data samples) both tensors of shape (N, ...).
        noise_samples: tensor of same shape as parameter samples.
        model: trainable pytorch model mapping from dimensions (n_param + 1 + n_data) to n_param.
        flow_matching: TorchCFM flow matching object, defaults to ConditionalFlowMatcher.
        batch_size: int, defaults to 256.

    Returns: trained model
    """
    N = noise_samples.shape[0]
    param_dim = noise_samples.shape[1]
    try:
        data_dim = joint_samples[1].shape[1]
    except IndexError:
        data_dim = 1

    if model is None:
        # large default model if no model specified
        model_ = torch.nn.Sequential(
            torch.nn.Linear(param_dim + 1 + data_dim, 1024),  # accepts 3 inputs: theta, t, x
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, param_dim),
        )
    else:
        model_ = model
    print("Training the following model:")
    print(model_)

    optimizer = torch.optim.Adam(model_.parameters())

    if flow_matching is None:
        fm = ConditionalFlowMatcher()
    else:
        fm = flow_matching

    n_batches = np.ceil(N/batch_size).astype(int)
    for k in range(n_batches):
        optimizer.zero_grad()

        theta_0 = noise_samples[k*batch_size:min((k+1)*batch_size, N),:]
        batch_size_k = theta_0.shape[0]

        theta_1, x = joint_samples
        theta_1 = theta_1[k*batch_size:min((k+1)*batch_size, N),:]
        x = x[k*batch_size:min((k+1)*batch_size, N),:]

        t = torch.rand(theta_0.shape[0]).type_as(theta_1)

        _, theta_t, ut = fm.sample_location_and_conditional_flow(x0=theta_0,
                                                                 x1=theta_1, t=t)  # conditional path

        theta_t = torch.reshape(theta_t, (batch_size_k, param_dim)).type(torch.float32)
        ut = torch.reshape(ut, (batch_size_k, param_dim)).type(torch.float32)
        x = torch.reshape(x, (batch_size_k, data_dim)).type(torch.float32)
        t = torch.reshape(t, (batch_size_k, 1)).type(torch.float32)

        vt = model_(torch.cat([theta_t, t, x], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        # backpropagation
        loss.backward()
        optimizer.step()

        if k % np.floor(n_batches/10) == 0:
            print(f"loss {loss.item():0.3f}")

    print(f"final loss {loss.item():0.3f}")

    return model_


def infer_FMPE_from_model(noise_samples, data, model,
                          n_steps=100):
    """

    Args:
        noise_samples: shape (N, param_dim)
        data: shape (1, data_dim)  TODO: how to incorporate multiple datapoints?
        model: trained FMPE model

    Returns:

    """
    M = noise_samples.shape[0]
    try:
        data_dim = data.shape[1]
    except AttributeError:
        data_dim = 1

    v_data = torch.reshape(torch.tensor([data for _ in range(M)]),
                          shape=(M, data_dim)).type(torch.float32)

    # fixing data input in model
    data_cond_model = DataConditionalModel(model, v_data)

    # NeuralODE object propagates particles according to trajectory defined by model
    node = NeuralODE(
        torch_wrapper(data_cond_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    # generating and propagating particles
    with torch.no_grad():  # disabling gradient computation, unnecessary during inference
        # node.forward passes samples through trajectory field defined by trained model
        posterior_samples = node.forward(noise_samples,
                                         t_span=torch.linspace(0, 1, n_steps))

    return posterior_samples
