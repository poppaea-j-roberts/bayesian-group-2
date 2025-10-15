""" example of applying flow matching posterior estimation (FMPE) to 1D bayesian inference """

from scipy import stats
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *



def sample_prior(mu_prior=0., sig_prior=1., size=1, seed=None):
    """ samples from a normal prior """
    return torch.tensor(stats.norm.rvs(mu_prior, sig_prior, size=size, random_state=seed))

def sample_joint(mu_prior=0., sig_prior=1., sig_data=1., size=1, seed=None):
    """
    samples from a simple normal prior for theta, and normal data likelihood with mean given by theta
    Args:
        mu_prior: mean for normal prior
        sig_prior: standard deviation for normal prior
        sig_data: standard deviation for conditional likelihood
        size: number of samples
    Returns: (theta, x) drawn from joint
    """
    theta = stats.norm.rvs(mu_prior, sig_prior, size=size, random_state=seed)
    x = np.array([stats.norm.rvs(theta[i], sig_data) for i in range(size)])

    return torch.tensor(theta), torch.tensor(x)

def exact_posterior(theta, x, mu_prior=0., sig_prior=1., sig_data=1.):
    """ exact theoretical posterior for inference specified by `sample_joint` """
    return stats.norm.pdf(theta, loc = (x*sig_prior**2 + mu_prior*sig_data**2)/(sig_prior**2 +sig_data**2),
                          scale = sig_prior*sig_data/np.sqrt(sig_prior**2+sig_data**2))


# defining  parameters
mu_prior = 0. # mean of prior on theta
sig_prior = 3. # std of prior on theta
sig_data = 1. # std of conditional likelihood data | theta

# TODO: extend to higher dimension examples
data_dim = 1
param_dim = 1

# defining model
class DataConditionalModel(torch.nn.Module):
    """ for an input model, fixes last n input values at data
    in FMPE setting, equivalent to conditioning a model on fixed data following training
    in practice, used like a separate model which operates on fewer inputs """
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def forward(self, theta):
        return self.model(torch.cat([theta, self.data], dim=1))

# example neural network <---- TODO: experiment with different architectures?
net = torch.nn.Sequential(
    torch.nn.Linear(param_dim + 1+ data_dim,32),  # accepts 3 inputs: theta, t, x
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, param_dim),
)


""" training """
optimizer = torch.optim.Adam(net.parameters())
batch_size = 256
n_batches = 2400

# ConditionalFlowMatcher object defines conditional paths, i.e. how data is mapped to noise distribution
# see eqns. (14) and (15) of A. Tong et al., Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport
FM = ConditionalFlowMatcher()

# loop training over batches
for k in range(n_batches):
    optimizer.zero_grad()

    # drawing samples from the prior - this is the 'noise' distribution
    theta_0 = sample_prior(mu_prior=mu_prior, sig_prior=sig_prior, size=batch_size)

    # drawing pairs (theta, x) from the joint, by sampling first theta~prior followed by x~p(x|theta)
    samples = sample_joint(size=batch_size, mu_prior=mu_prior,sig_prior=sig_prior, sig_data=sig_data)
    theta_1, x = samples

    # uniform random timepoint
    t = torch.rand(theta_0.shape[0]).type_as(theta_1)

    # interpolating intermediate point and computing conditional flow (see details of ConditionalFlowMatcher)
    # theta_t = t*theta_1 + (1-t)*theta_0, ut = theta_1 - theta_t
    _, theta_t, ut = FM.sample_location_and_conditional_flow(x0=theta_0,
                                                        x1=theta_1, t=t) # conditional path

    # reformating vectors for model
    theta_t = torch.reshape(theta_t, (batch_size, param_dim)).type(torch.float32)
    ut = torch.reshape(ut, (batch_size, param_dim)).type(torch.float32)
    x = torch.reshape(x, (batch_size, data_dim)).type(torch.float32)
    t = torch.reshape(t, (batch_size, 1)).type(torch.float32)

    # applying  model to theta, t, x
    vt = net(torch.cat([theta_t, t, x],dim=-1))
    loss= torch.mean((vt-ut)**2)

    # backpropagation
    loss.backward()
    optimizer.step()

    if k%200==0:
        print(f"loss {loss.item():0.3f}")


""" inference """
M = 5000 # number of trajectories to sample
obs = 3. # dummy data

# vector of fixed obs
v_obs = torch.reshape(torch.tensor([obs for _ in range(M)]),
                      shape=(M, data_dim)).type(torch.float32)

# fixing data input in model
data_cond_model = DataConditionalModel(net, v_obs)

# NeuralODE object propagates particles according to trajectory defined by model
node = NeuralODE(
    torch_wrapper(data_cond_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
)

# generating and propagating particles
with torch.no_grad(): # disabling gradient computation, unnecessary during inference
    # sampling from prior
    theta_0 = torch.reshape(torch.tensor(stats.norm.rvs(size=M,
                                                        scale=sig_prior)),
                            shape=(M,param_dim)).type(torch.float32)

    # node.forward passes samples through trajectory field defined by trained model
    posterior_samples = node.forward(theta_0,
                                    t_span=torch.linspace(0, 1, 100))


""" generating gif of distribution trajectory"""
from matplotlib.animation import FuncAnimation

data = posterior_samples[1][:,:,0]
T = data.shape[0]
n_bins = 120
x_min, x_max = -4, 8
y_min, y_max = 0, 0.7

fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

hist = ax.hist(data[0,:], bins=n_bins, density=True)

def update(frame):
    if frame<75:    # show prior distribution for 75 frames
        ax.plot(np.linspace(x_min, x_max, 250),
                stats.norm.pdf(np.linspace(x_min, x_max, 250), scale=sig_prior))
        ax.set_title("Samples from prior")

    if 175 > frame >= 75:   # shows intermediate histogram, corresponding to time (1 - (frame-75)/T)
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"time = {(frame-75)/T}")
        ax.hist(data[frame-75,:], bins=n_bins, density=True)

    if frame>=175:  # show final distribution for 75 frames
        ax.plot(np.linspace(x_min, x_max, 250),
                exact_posterior(np.linspace(x_min, x_max, 250), x=obs, sig_prior=sig_prior))
        ax.set_title("Transformed samples vs exact posterior")

ani = FuncAnimation(fig, update, frames=T+75+75, interval=50, blit=False)
ani.save("fm-hist-animation.gif", writer='pillow', fps=15)
plt.show()