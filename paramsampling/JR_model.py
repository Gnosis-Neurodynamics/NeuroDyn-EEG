import numpy as np


def sigmoid(x, rmax=1, beta=1, x0=0):
    return rmax / (1 + np.exp(beta * (x0 - x)))


def check_type(data, shape=(90,)):
    if np.isscalar(data):
        return np.zeros(shape) + data
    else:
        return data


class JansenRit:
    def __init__(
        self,
        dt=0.0001,
        seed=42,
        nonlinearity=sigmoid,
        tau_exc=1 / 50,
        tau_inh=1 / 50,
        theta=6,
        beta=0.56,
        rmax=5,
        C1=1,
        C2=0.8,
        C3=0.25,
        C4=0.25,
    ):

        n_synapses = 135  # number of synapses between populations
        self.C_1 = (
            C1 * n_synapses
        )  # probability of connection between excitatory and pyramidal populations
        self.C_2 = C2 * n_synapses
        self.C_3 = C3 * n_synapses
        self.C_4 = C4 * n_synapses
        self.tau_exc = tau_exc  # time scale for excitatory population ~10ms
        self.tau_inh = tau_inh  # time scale for inhibitory population ~20ms
        self.G_exc = 3.25  # average excitatory synaptic gain (mV)
        self.G_inh = 22  # average inhibitory synaptic gain
        self.rmax = rmax  # amplitude of sigmoid in Hz (max firing rate)
        self.beta = beta  # slope of sigmoid (mV^-1)
        self.theta = theta  # threshold of sigmoid (mV)
        self.v = 10  # conduction velocity
        self.P = 150  # external input to each of the neural masses
        # self.Coupling = 0.1 # coupling between the neural masses (global coupling strength)
        self.nstates = 6  # number of state variables
        self.nosie = np.random.normal
        self.dt = dt
        self.seed = seed

        self.x = np.zeros((6,))  # state of the network
        self.S = lambda x: nonlinearity(
            x, rmax=self.rmax, beta=self.beta, x0=self.theta
        )  # nonlinearity function

    def step(self, I=0, Ip=0, Ii=0):
        """
        Compute one step of the Jansen-Rit model.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        x0, x1, x2, xdot0, xdot1, xdot2 = self.x  # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(np.asarray([x1 - x2, self.C_1 * x0, self.C_3 * x0]))
        input_excitatory = (
            self.C_2 * firing_rates[1] + I + Ip
        )  # contribution from other nodes will go here
        xdot0_next = (
            xdot0
            + self.dt
            * (
                self.G_exc * 1.0 * (firing_rates[0] + Ii)
                - 2 * xdot0
                - x0 / self.tau_exc
            )
            / self.tau_exc
        )  # pyramidal cell
        xdot1_next = (
            xdot1
            + self.dt
            * (self.G_exc * input_excitatory - 2 * xdot1 - x1 / self.tau_exc)
            / self.tau_exc
        )  # excitatory stellate cell
        xdot2_next = (
            xdot2
            + self.dt
            * (
                self.G_inh * (self.C_4 * firing_rates[2] + Ip)
                - 2 * xdot2
                - x2 / self.tau_inh
            )
            / self.tau_inh
        )  # inhibitory interneuron
        x0_next = x0 + xdot0 * self.dt
        x1_next = x1 + xdot1 * self.dt
        x2_next = x2 + xdot2 * self.dt
        self.x = np.array(
            [x0_next, x1_next, x2_next, xdot0_next, xdot1_next, xdot2_next]
        )

    def read_out(self):
        return self.x[1] - self.x[2]

    def simulate(self, x0, tmax=1, noise=0.0, P=None, Ip=None, Ii=None):
        """
        Simulate the Jansen-Rit model and monitor the output.

        Parameters
        ----------
        x0 : array_like
            The initial state of the system. Shape (N,).
        tmax : float
            The maximum time to simulate.
        noise : float
            The standard deviation of the noise to add to the system.

        Returns
        -------
        x : array_like
            The simulated time series. Shape (n, N).
        """
        # rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, 6))
        o = np.zeros((n, 1))
        self.x = x0
        x[0] = x0
        o[0] = self.read_out()
        # dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            self.step(
                I=self.P if P is None else P[i],
                Ip=0 if Ip is None else Ip[i],
                Ii=0 if Ii is None else Ii[i],
            )
            # noise = rng.standard_normal(size=self.x.shape) * dt_noise
            # print(noise.shape)
            x[i] = self.x  # + noise
            o[i] = self.read_out()
        return x, o


class JansenRitExtended:
    """
    Jansen-Rit model - Exctended version: dual kinetic model.

    See https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00924/1-s2.0-S1053811903004579/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDklbniEg%2BIxtbMhqre0GLXBUY61F7QwRGlTcS0Tw50mAIgPijMAG0JRY86ILcJl5khJAbWRSGrqqz8rRNKkeVXhtMqvAUI6v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDDWGjReDtp%2FZt%2BvT4SqQBUgH%2FBzAz1ffqLsZXij%2Fhvyo540aF9iKjI5qqwn%2FClzZq1dP%2BofFqhpbLq%2FbRaP98Et%2Bs5VZoAoNHhC8dMDRlrxY47ZuTWehsc2C8ZUcz6D9lYArh5ggiCQZMm9040OWVNDarLG8631K4g0HFpEHwsubfZoIUgs5XABH%2FyF1NE2zXo3JXhU%2FKwsZHVqqidrV0nv%2B9IJ5%2BigTmMVePAINzRUuQjnFgbyqNvqMwUsNDi92QSN7u%2BdJi0ksGwqTWyBCM7MUrwK%2FisZjWmQcQeCx%2FuLyhE77tU6x7gpR%2BGdE%2BOmiISZDROqxwq%2FmsC%2B%2FR%2BMsFubuDhdTI8n8kllkh079IhICKgEzhzFLWgmNxXdbxBS6vQi13IA5SNlQuuzArMu0Z1GkM9EwbXRUI3u4oeOMtGOslgCYdgsSJpebcg8zOg2ueLBgHSfubsGGJ5l5SxOR%2FIVF6tHjzLoCz2njmpmtSr%2Bmi56T9qKmkUC4sAoYnbpllz3yd%2F4%2B2Xh3AruYCVWyJ%2FVLgTbrPXYpihKXryoL1DNw94iDb%2FWutOEqh46F64lPV96HjCvNME4smEYyyPKz8DWYdMcU6U35qVbk%2BXlv%2FoDv7AIjCJi3l9U98IL%2BiZMAnCN8u88BtwF8o7PrMAMfRzjQl53oJEt%2BCfnIMZhwy0raTnU3Sb61tmgBqoIrY7NFmWjrAmqNJilwO16T7RIJkjRhufHocRzquTEj6F5NyPrir%2FeVwlg3%2BM8kgmsdsiIfLR9tkvkXCcSx1M4%2FwdK9id37RAh5D%2Fcw0jAcqEinfgj%2Bw4Z9MX5FU91B9pOJIBk%2BGRraBuxSsxznjeQ%2BlOZVwtQ7nVwE%2Bk0XUHkMv075s9RxWPIrg48wrQMqJUhLaW7zMK%2B%2B7KoGOrEBKbm0kOWBELm7BGve3m4Vj6De%2BonWolZfqid1yachKSVCGGly07GpJptQJqCsQftOtoRw966JD257GoZ%2BssIo%2FrdAhYGJXSWfvpbzGVT3etLIjbrApoBoQMzw%2BVTy1I%2FyomAtv2yXcop%2FNGlQyOkREmdcc3cbpVi78%2FPiChzyI3ESzp4hwWnNKs2cFoQE15W8GGT0FxkBk0QoXJN2MiPdAS%2Fd7Nrw4MOMPaseHhIfQx81&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231120T101219Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXMTQIC7Q%2F20231120%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=29cd9a0fade336c06381b528450f2b797b8b67eaa9ecc3e029ceb182ac9dcf40&hash=d99e790b5d5df95dc08507ca2341780d14b640ae4c5e57d1b6e1cac92280cb90&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903004579&tid=spdf-2465ea3f-65c5-4f0d-9fd9-eaf43d018e8d&sid=3f6dcc697a1fe84db69a8074585332d50fa6gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f57525b01065d0459&rr=828fe9f6bfba5c4b&cc=nl

    We model two parallel subpopulations with different kinematics in order to capture multiband or broadband dynamics.
    """

    def __init__(
        self,
        w=0.5,
        dt=0.0001,
        seed=42,
        nonlinearity=sigmoid,
        tau_exc1=1 / 100,
        tau_inh1=1 / 50,
        tau_exc2=0.0046,
        tau_inh2=0.0029,
        rmax=5,
        beta=0.56,
        theta=6,
        C1=1,
        C2=0.8,
        C3=0.25,
        C4=0.25,
    ):

        # ~ 10 Hz dynamics
        self.tau_exc_1 = tau_exc1  # time scale for excitatory population ~10ms
        self.tau_inh_1 = tau_inh1  # time scale for inhibitory population ~20ms
        # ~ 43 Hz dynamics
        self.tau_exc_2 = tau_exc2  # time scale for excitatory population ~4.6ms
        self.tau_inh_2 = tau_inh2  # time scale for inhibitory population ~2.9ms
        self.G_exc_1 = 3.25  # average excitatory synaptic gain (mV)# 3.25
        self.G_inh_1 = 22  # average inhibitory synaptic gain # 22
        self.G_exc_2 = 2 * 3.25  # average excitatory synaptic gain (mV) 2 * 3.25
        self.G_inh_2 = 150  # average inhibitory synaptic gain prigin 150
        self.w = w  # relative contribution of the first subpopulation

        # The rest is the same as the Jansen-Rit model
        n_synapses = 135  # number of synapses between populations
        self.C_1 = (
            C1 * n_synapses
        )  # probability of connection between excitatory and pyramidal populations
        self.C_2 = C2 * n_synapses
        self.C_3 = C3 * n_synapses
        self.C_4 = C4 * n_synapses
        self.rmax = rmax  # tau_exc1 * rmax / 0.02 # rmax # amplitude of sigmoid in Hz (max firing rate) # 5
        self.beta = (
            beta  # tau_exc1 * beta / 0.02 # beta # slope of sigmoid (mV^-1) # 0.56
        )
        self.theta = (
            theta  # tau_exc1 * theta / 0.02 # theta # threshold of sigmoid (mV) # 6
        )
        self.v = 10  # conduction velocity
        self.P = 150  # external input to each of the neural masses
        # self.Coupling = 0.1 # coupling between the neural masses (global coupling strength)

        self.x = np.zeros((2 * 6,))
        # np.zeros((2 * 6,)) # state of the network
        self.nstates = 2 * 6
        self.S = lambda x: nonlinearity(
            x, rmax=self.rmax, beta=self.beta, x0=self.theta
        )  # nonlinearity function

    def step(self, I=0.0):
        """
        Compute one step of the Jansen-Rit model.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        (
            x0_1,
            x1_1,
            x2_1,
            xdot0_1,
            xdot1_1,
            xdot2_1,
            x0_2,
            x1_2,
            x2_2,
            xdot0_2,
            xdot1_2,
            xdot2_2,
        ) = self.x  # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(
            np.asarray(
                [
                    self.w * (x1_1 - x2_1) + (1 - self.w) * (x1_2 - x2_2),
                    self.C_1 * (self.w * x0_1 + (1 - self.w) * x0_2),
                    self.C_3 * (self.w * x0_1 + (1 - self.w) * x0_2),
                ]
            )
        )
        input_excitatory = (
            self.C_2 * firing_rates[1] + I
        )  # contribution from other nodes will go here
        # pop 1
        xdot0_next_1 = (
            xdot0_1
            + self.dt
            * (
                self.G_exc_1 * 1.0 * firing_rates[0]
                - 2 * xdot0_1
                - x0_1 / self.tau_exc_1
            )
            / self.tau_exc_1
        )
        xdot1_next_1 = (
            xdot1_1
            + self.dt
            * (self.G_exc_1 * input_excitatory - 2 * xdot1_1 - x1_1 / self.tau_exc_1)
            / self.tau_exc_1
        )
        xdot2_next_1 = (
            xdot2_1
            + self.dt
            * (
                self.G_inh_1 * self.C_4 * firing_rates[2]
                - 2 * xdot2_1
                - x2_1 / self.tau_inh_1
            )
            / self.tau_inh_1
        )
        x0_next_1 = x0_1 + xdot0_1 * self.dt
        x1_next_1 = x1_1 + xdot1_1 * self.dt
        x2_next_1 = x2_1 + xdot2_1 * self.dt
        # Pop 2
        xdot0_next_2 = (
            xdot0_2
            + self.dt
            * (
                self.G_exc_2 * 1.0 * firing_rates[0]
                - 2 * xdot0_2
                - x0_2 / self.tau_exc_2
            )
            / self.tau_exc_2
        )
        xdot1_next_2 = (
            xdot1_2
            + self.dt
            * (self.G_exc_2 * input_excitatory - 2 * xdot1_2 - x1_2 / self.tau_exc_2)
            / self.tau_exc_2
        )
        xdot2_next_2 = (
            xdot2_2
            + self.dt
            * (
                self.G_inh_2 * self.C_4 * firing_rates[2]
                - 2 * xdot2_2
                - x2_2 / self.tau_inh_2
            )
            / self.tau_inh_2
        )
        x0_next_2 = x0_2 + xdot0_2 * self.dt
        x1_next_2 = x1_2 + xdot1_2 * self.dt
        x2_next_2 = x2_2 + xdot2_2 * self.dt
        self.x = np.array(
            [
                x0_next_1,
                x1_next_1,
                x2_next_1,
                xdot0_next_1,
                xdot1_next_1,
                xdot2_next_1,
                x0_next_2,
                x1_next_2,
                x2_next_2,
                xdot0_next_2,
                xdot1_next_2,
                xdot2_next_2,
            ]
        )

    def read_out(self):
        return self.w * (self.x[1] - self.x[2]) + (1 - self.w) * (self.x[7] - self.x[8])

    def simulate(self, x0, tmax=1, noise=0.0, P=None):

        rng = np.random.default_rng(self.seed)
        n = int(tmax / self.dt)
        x = np.zeros((n, 12))
        o = np.zeros((n, 1))
        self.x = x0
        x[0] = x0
        o[0] = self.read_out()
        dt_noise = np.sqrt(self.dt) * noise
        for i in range(1, n):
            self.step(I=self.P if P is None else P[i])
            # noise = rng.standard_normal(size=self.x.shape) * dt_noise
            x[i] = self.x
            o[i] = self.read_out()
        return x, o


class JRNetwork:
    def __init__(
        self,
        N=2,
        W=np.asarray([[0, 1], [0, 0]]),
        delay=0.01,
        band_w=0.5,
        node_mode="single",
        dt=0.001,
        seed=42,
        tau_exc1=50,
        tau_inh1=50,
        tau_exc2=0.0046,
        tau_inh2=0.0029,
        theta=6,
        beta=0.56,
        rmax=5,
        C1=1,
        C2=0.8,
        C3=0.25,
        C4=0.25,
    ):

        self.rng = np.random.default_rng(seed)
        self.N = N  # number of neurons/nodes
        self.W = W  # connectivity matrix (W_ij is the connection from i to j, between 0 and 1, relative contribution)
        self.K = (
            W.copy()
        )  # updated connectivity in case of normalisation by activity std
        self.delay = delay  # delay (10ms)
        self.dt = dt  # sampling rate
        self.seed = seed  # random seed
        if node_mode != "single":
            if np.isscalar(
                band_w
            ):  # if w is a scalar, then it is the same for all nodes
                band_w = band_w * np.ones((self.N,))
            self.nodes = [
                JansenRitExtended(
                    w=band_w[k],
                    dt=dt,
                    seed=42,
                    theta=theta,
                    rmax=rmax,
                    beta=beta,
                    C1=C1,
                    C2=C2,
                    C3=C3,
                    C4=C4,  # self.rng.integers(k+seed),
                    tau_exc1=tau_exc1[k],
                    tau_inh1=tau_inh1[k],
                    tau_exc2=tau_exc2[k],
                    tau_inh2=tau_inh2[k],
                )
                for k in range(N)
            ]  # get different systems/rng for each node
        else:
            self.nodes = [
                JansenRit(
                    dt=dt,
                    seed=42,
                    theta=check_type(theta, shape=(self.N))[k],
                    rmax=check_type(rmax, shape=(self.N))[k],
                    beta=check_type(beta, shape=(self.N))[k],
                    C1=check_type(C1, shape=(self.N))[k],
                    C2=check_type(C2, shape=(self.N))[k],
                    C3=check_type(C3, shape=(self.N))[k],
                    C4=check_type(C4, shape=(self.N))[k],  # self.rng.integers(k+seed),
                    tau_exc=tau_exc1[k],
                    tau_inh=tau_inh1[k],
                )
                for k in range(N)
            ]
        self.S = self.nodes[0].S  # nonlinearity function
        # self.delayed_states = np.zeros((N, self.nodes[0].nstates)) # delayed states of the nodes (state of the nodes at t-dt)
        self.delayed_states = np.zeros(
            (N, 1)
        )  # delayed states of the nodes (state of the nodes at t-dt) / readout only
        self.init_states = np.zeros(shape=(self.N,)) + 5
        # self.init_states = np.random.normal(loc=5, scale=0.1, size=((self.N,)))

    def update_connectivity(self, x, sigma_p=1):
        """
        x represents the firing rates output of all nodes, needed to compute the standard deviation
        Shape of x: (N, ntimes)
        """
        # See ref (David & Friston 2004: A Neural Mass model for M/EEG: coupling and neuronal dynamics)
        if np.isscalar(
            sigma_p
        ):  # if sigma_p is a scalar, then it is the same for all nodes
            sigma_p = sigma_p * np.ones((self.N,))

        if (
            np.ndim(x) <= 1 or x.shape[1] <= 1
        ):  # or np.isnan(x).any()  or np.isinf(x).any():
            # sigma_rate = np.random.normal(loc=100, scale=10, size=(self.N))
            # print("init mean 100 std 10")
            sigma_rate = np.ones((self.N,))
            # print("init 1")
        else:
            sigma_rate = np.std(x, axis=1)

        self.K = (
            sigma_p[np.newaxis, :]
            * np.sqrt(2 * self.W - self.W**2)
            / sigma_rate[:, np.newaxis]
        )

    def simulate(self, tmax=1, P=220, sigma_p=22):
        t = np.arange(0, tmax, self.dt)
        tdelay, kdelay = 0, 0
        outs = [self.init_states]
        # external, interarea = [], []

        for k, tt in enumerate(t):
            update_states = self.step(
                P=P[k] if not isinstance(P, (int, float)) else P,
                sigma_p=sigma_p,
                history_outs=self.S(np.asarray(outs).T),
            )
            outs.append(update_states)
            # external.append(update_external)
            # interarea.append(update_interarea)

            tdelay += self.dt
            kdelay += 1
            if tdelay >= self.delay:
                self.delayed_states = outs[-kdelay]
                tdelay, kdelay = 0, 0

            # if k % (len(t) // 10) == 0:
            #     print(np.round(np.array([self.K.max(), self.K.min(), self.K.mean(), self.W.max(), self.W.min(), self.W.mean()]), 2))
        return np.asarray(outs)  # , np.array(external), np.array(interarea)

    def step(self, P=220, sigma_p=22, history_outs=None):
        if np.isscalar(
            sigma_p
        ):  # if sigma_p is a scalar, then it is the same for all nodes
            sigma_p = sigma_p * np.ones((self.N,))

        self.update_connectivity(
            history_outs, sigma_p=sigma_p
        )  # update connectivity based on the history of the outputs
        # external_input_fluctuation = (sigma_p * self.rng.standard_normal(size=(self.N,))) * (1 - self.W.sum(axis=0))
        # mean firing rate = mean S pyc
        mean_firing = history_outs.mean(axis=-1)  # self.S(history_outs).mean(axis=-1)
        interarea_contributions = (
            self.S(self.delayed_states).ravel() - mean_firing
        ) @ self.K  # using normalised connectivity a origin 3.84
        outs = []

        for k, n in enumerate(self.nodes):
            # Below y_1 and y_2 corresponds to the fast and slow dynamics respectively of the extended Jansen-Rit model
            #  I = p + (1-k21) * p̃ + k21_star *(S(w*y_1(t - δ) + (1-w)*y_2(t - δ)) - a) where "a" is the mean firing rate
            # we remove the mean firing rate to ensure mean input is conserved with different coupling strengths
            # a = 3.5 in the reference paper, my measured mean is more around 3.84
            n.step(
                I=P
                +
                #    external_input_fluctuation[k]  + # noise input scaled : (1-k21) * p̃
                interarea_contributions[
                    k
                ]  # k21_star * (S(w*y_1(t - δ) + (1-w)*y_2(t - δ)) - a)
            )  # the input to each node is the output of all the other nodes
            outs.append(n.read_out())
            # for i in range(self.N):
            #     I += self.K[i, n] * (outs[i] - 3.84)

        return np.asarray(outs)  # , external_input_fluctuation, interarea_contributions

    def reset(self):
        self.delayed_states = np.zeros((self.N, 1))
        for n in self.nodes:
            n.x = np.zeros((n.nstates,))
        self.K = self.W.copy()


class JRNetwork_optim:
    def __init__(
        self,
        N=90,
        W=np.array([[0, 1], [1, 0]]),
        delay=0.01,
        band_w=0.5,
        node_mode="single",
        dt=0.001,
        seed=42,
        tau_exc1=0.015,
        tau_inh1=0.012,
        tau_exc2=0.015,
        tau_inh2=0.012,
        theta=6,
        beta=0.56,
        rmax=5,
        C1=1,
        C2=0.8,
        C3=0.25,
        C4=0.25,
    ):
        self.N = N
        self.W = W
        self.K = W.copy()
        self.delay = delay
        self.dt = dt
        self.seed = seed
        self.node_mode = node_mode
        self.tau_exc1 = check_type(tau_exc1, shape=(self.N,))
        self.tau_inh1 = check_type(tau_inh1, shape=(self.N,))

        self.tau_exc2 = check_type(tau_exc2, shape=(self.N,))
        self.tau_inh2 = check_type(tau_inh2, shape=(self.N,))
        n_synapses = 135
        self.C1 = check_type(C1, shape=(self.N,)) * n_synapses
        self.C2 = check_type(C2, shape=(self.N,)) * n_synapses
        self.C3 = check_type(C3, shape=(self.N,)) * n_synapses
        self.C4 = check_type(C4, shape=(self.N,)) * n_synapses

        self.theta = check_type(theta, shape=(self.N,))
        self.beta = check_type(beta, shape=(self.N,))
        self.rmax = check_type(rmax, shape=(self.N,))

        self.G_exc1 = 3.25
        self.G_inh1 = 22

        self.G_exc2 = 32.5 / (1000 * self.tau_exc2)
        self.G_inh2 = 440 / (1000 * self.tau_inh2)

        self.S = lambda x: sigmoid(x, rmax=self.rmax, beta=self.beta, x0=self.theta)

        if node_mode == "single":
            self.nodes = np.zeros(shape=(6, N))

        else:
            self.nodes = np.zeros(shape=(12, N))
            self.band_w = check_type(band_w, shape=(self.N,))

        self.init_states = np.zeros(shape=(self.N,)) + 5
        self.x = self.nodes.copy()
        self.delay_states = np.zeros((self.N,))

    def update_connectivity(self, x, sigma_p=22):
        if np.ndim(x) <= 1 or x.shape[1] <= 1:
            sigma_rate = np.ones(shape=(self.N, 1))
        else:
            sigma_rate = np.std(x, axis=-1, keepdims=True)

        self.K = sigma_p * np.sqrt(2 * self.W - self.W**2) / sigma_rate

    def step_signale_node(self, I=0, Ip=0, Ii=0):
        """
        Compute one step of the Jansen-Rit model.
        """
        # 0: pyramidal, 1: excitatory, 2: inhibitory
        x0, x1, x2, xdot0, xdot1, xdot2 = self.x  # unpack the state
        # Input received by each population
        # x1 - x2: difference between excitatory and inhibitory activity, which is the input received by the pyramidal population interpreted as the average potential of pyramidal populations
        # self.C_1 * x0: input received by the excitatory population
        # self.C_3 * x0: input received by the inhibitory population
        firing_rates = self.S(np.asarray([x1 - x2, self.C1 * x0, self.C3 * x0]))
        input_excitatory = (
            self.C2 * firing_rates[1] + I + Ip
        )  # contribution from other nodes will go here
        xdot0_next = (
            xdot0
            + self.dt
            * (
                self.G_exc1 * 1.0 * (firing_rates[0] + Ii)
                - 2 * xdot0
                - x0 / self.tau_exc1
            )
            / self.tau_exc1
        )  # pyramidal cell
        xdot1_next = (
            xdot1
            + self.dt
            * (self.G_exc1 * input_excitatory - 2 * xdot1 - x1 / self.tau_exc1)
            / self.tau_exc1
        )  # excitatory stellate cell
        xdot2_next = (
            xdot2
            + self.dt
            * (
                self.G_inh1 * (self.C4 * firing_rates[2] + Ip)
                - 2 * xdot2
                - x2 / self.tau_inh1
            )
            / self.tau_inh1
        )  # inhibitory interneuron
        x0_next = x0 + xdot0 * self.dt
        x1_next = x1 + xdot1 * self.dt
        x2_next = x2 + xdot2 * self.dt
        self.x = np.array(
            [x0_next, x1_next, x2_next, xdot0_next, xdot1_next, xdot2_next]
        )

    def step_fastslow_node(self, I=0):

        (
            x0_1,
            x1_1,
            x2_1,
            xdot0_1,
            xdot1_1,
            xdot2_1,
            x0_2,
            x1_2,
            x2_2,
            xdot0_2,
            xdot1_2,
            xdot2_2,
        ) = self.x  # unpack the state

        firing_rates = self.S(
            np.asarray(
                [
                    self.band_w * (x1_1 - x2_1) + (1 - self.band_w) * (x1_2 - x2_2),
                    self.C1 * (self.band_w * x0_1 + (1 - self.band_w) * x0_2),
                    self.C3 * (self.band_w * x0_1 + (1 - self.band_w) * x0_2),
                ]
            )
        )
        input_excitatory = (
            self.C2 * firing_rates[1] + I
        )  # contribution from other nodes will go here
        # pop 1
        xdot0_next_1 = (
            xdot0_1
            + self.dt
            * (self.G_exc1 * 1.0 * firing_rates[0] - 2 * xdot0_1 - x0_1 / self.tau_exc1)
            / self.tau_exc1
        )
        xdot1_next_1 = (
            xdot1_1
            + self.dt
            * (self.G_exc1 * input_excitatory - 2 * xdot1_1 - x1_1 / self.tau_exc1)
            / self.tau_exc1
        )
        xdot2_next_1 = (
            xdot2_1
            + self.dt
            * (
                self.G_inh1 * self.C4 * firing_rates[2]
                - 2 * xdot2_1
                - x2_1 / self.tau_inh1
            )
            / self.tau_inh1
        )
        x0_next_1 = x0_1 + xdot0_1 * self.dt
        x1_next_1 = x1_1 + xdot1_1 * self.dt
        x2_next_1 = x2_1 + xdot2_1 * self.dt
        # Pop 2
        xdot0_next_2 = (
            xdot0_2
            + self.dt
            * (self.G_exc2 * 1.0 * firing_rates[0] - 2 * xdot0_2 - x0_2 / self.tau_exc2)
            / self.tau_exc2
        )
        xdot1_next_2 = (
            xdot1_2
            + self.dt
            * (self.G_exc2 * input_excitatory - 2 * xdot1_2 - x1_2 / self.tau_exc2)
            / self.tau_exc2
        )
        xdot2_next_2 = (
            xdot2_2
            + self.dt
            * (
                self.G_inh2 * self.C4 * firing_rates[2]
                - 2 * xdot2_2
                - x2_2 / self.tau_inh2
            )
            / self.tau_inh2
        )
        x0_next_2 = x0_2 + xdot0_2 * self.dt
        x1_next_2 = x1_2 + xdot1_2 * self.dt
        x2_next_2 = x2_2 + xdot2_2 * self.dt
        self.x = np.array(
            [
                x0_next_1,
                x1_next_1,
                x2_next_1,
                xdot0_next_1,
                xdot1_next_1,
                xdot2_next_1,
                x0_next_2,
                x1_next_2,
                x2_next_2,
                xdot0_next_2,
                xdot1_next_2,
                xdot2_next_2,
            ]
        )

    def read_out_single_node(
        self,
    ):
        res = self.x[1] - self.x[2]
        return res

    def read_out_fastslow_node(self):
        res = self.band_w * (self.x[1] - self.x[2]) + (1 - self.band_w) * (
            self.x[7] - self.x[8]
        )
        return res

    def simulate(self, tmax, P=220, sigma_p=22):
        t = np.arange(0, tmax, self.dt)
        tdelay, kdelay = 0, 0

        outs = [self.init_states]

        sigma_p = check_type(sigma_p, shape=(self.N,))

        for k, tt in enumerate(t):

            history_outs = self.S(np.asarray(outs)).T
            self.update_connectivity(history_outs, sigma_p=sigma_p)

            mean_firing = history_outs.mean(axis=-1)

            interarea_contributions = (
                self.S(self.delay_states.squeeze()) - mean_firing
            ) @ self.K

            p = P[k] if not isinstance(P, (int, float)) else P

            if self.node_mode == "single":
                self.step_signale_node(I=interarea_contributions + p, Ip=0, Ii=0)

                outs.append(self.read_out_single_node())

            else:
                self.step_fastslow_node(I=interarea_contributions + p)
                outs.append(self.read_out_fastslow_node())

            tdelay += self.dt
            kdelay += 1

            if tdelay >= self.delay:
                self.delay_states = outs[-kdelay]
                tdelay, kdelay = 0, 0

        return np.asarray(outs)
