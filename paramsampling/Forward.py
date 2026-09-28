import pandas as pd
import mne
import numpy as np
import os, warnings

from eegaux.montage import STANDARD_1020, STANDARD_1005
from eegaux.preprocessing import set_data_RFN

from .JR_model import JRNetwork_optim
import scipy.io as sio
from mne.minimum_norm import make_inverse_operator, apply_inverse
from project_paths import resource_path, require_resource

fwd_base = resource_path("fsaverage_volume_forward_solution.fif")
atlas_match = resource_path("AAL_regions_MNI_90_MatchVolfwd.csv")

rename_1005 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
rename_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def standard_channels(
    feature_dict, S={"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}, ch_upper=False
):
    if isinstance(feature_dict, list):
        res = []
        for f in feature_dict:
            for s in S.keys():
                f = S[s] if f == s else f
            if ch_upper:
                res.append(f.upper())
            else:
                res.append(f)
        return res


def read_matched_source(pick_channels=STANDARD_1020):
    warnings.filterwarnings("ignore")
    aal_labels = pd.read_csv(require_resource(atlas_match))
    matched_index = aal_labels[["match_src_index"]].values.astype(int)
    fwd = mne.read_forward_solution(require_resource(fwd_base), verbose=False)
    fwd_channels = fwd.ch_names
    fwd_channels = standard_channels(fwd_channels, rename_1005, ch_upper=True)

    if pick_channels:
        pick_channels = [ch.upper() for ch in pick_channels]
        pick_id = [fwd_channels.index(ch) for ch in pick_channels]
    else:
        pick_id = np.arange(len(pick_channels))

    vetrno_id = fwd["src"][0]["vertno"]

    sol = fwd["sol"]["data"][:, vetrno_id]

    matched_atlas_sol = sol[:, matched_index[:, 0]]
    res = matched_atlas_sol[pick_id, :]
    return res


def inference_inv(info, pick_channels=STANDARD_1005, noise_cov=None):
    aal_labels = pd.read_csv(require_resource(atlas_match))
    matched_index = aal_labels[["match_src_index"]].values.astype(int)
    fwd = mne.read_forward_solution(require_resource(fwd_base), verbose=False)

    fwd_channels = fwd.ch_names
    fwd_channels = standard_channels(fwd_channels, rename_1005, ch_upper=True)
    pick_channels = [ch.replace("Z", "z").replace("FP", "Fp") for ch in pick_channels]
    noise_cov["projs"] = []
    inv = make_inverse_operator(
        info=info,
        forward=fwd,
        noise_cov=noise_cov,
        loose="auto",
        depth=None,
        verbose=False,
    )

    return inv


class NoiseGenerator(object):
    def __init__(self, fs=1000, duration=None):
        self.fs = fs
        self.dt = 1 / fs
        self.duration = duration
        self.n_points = int(duration / self.dt)

    def generate_white_noise(self, mean, std):
        return np.random.normal(mean, std, self.n_points)

    def generate_colored_Gaussiannoise(self, tau=0.001, mean=220, std=44):

        white_noise = self.generate_white_noise(mean=mean, std=std)

        alpha = np.exp(-self.dt / tau)
        colored_noise = np.zeros(self.n_points)

        colored_noise[0] = white_noise[0]
        for i in range(1, self.n_points):
            colored_noise[i] = (
                alpha * colored_noise[i - 1] + (1 - alpha) * white_noise[i]
            )

        return colored_noise

    def generate_ACsimulate(self, ac_f=10, ac_amp=10, start=1, stop=2):
        t_index = np.linspace(0, self.duration, self.fs * self.duration)
        out = np.zeros_like(t_index)

        duration = stop - start
        t = np.linspace(0, duration, int(duration * self.fs))

        ac_ = np.sin(2 * np.pi * ac_f * t) * ac_amp
        out[int(start * self.fs) : int(stop * self.fs)] = ac_
        # out = np.where(out < 0, out, 0)
        return out


class GenerateSimuSignal(object):
    def __init__(
        self,
        source_sfreq=1000,
        T=10,
        eeg_sfreq=256,
        eeg_filter=[1, 40],
        pick_channels=STANDARD_1020,
    ):
        mat_data = sio.loadmat(
            require_resource(resource_path("SCmatrices88healthy.mat"))
        )
        self.fiber_con = mat_data["SCmatrices"].mean(axis=0)
        self.nodes = self.fiber_con.shape[0]
        self.source_sfreq = source_sfreq
        self.simulate_times = T
        self.eeg_sfreq = eeg_sfreq
        self.eeg_filter = eeg_filter

        self.noise_worker = NoiseGenerator(
            fs=self.source_sfreq, duration=self.simulate_times
        )

        self.delay = 1 / self.source_sfreq
        self.amp_scale = 1e-3

        self.fwd_sol = read_matched_source(pick_channels)

    def load_noise(self, tau=0.001, noise_mean=220, noise_std=40):
        gau_noise = self.noise_worker.generate_colored_Gaussiannoise(
            tau=tau, mean=noise_mean, std=noise_std
        )
        return gau_noise

    def fit_singleband_JRnetwork(
        self,
        tau_e1,
        tau_i1,
        tau_e2,
        tau_i2,
        band_w=0.5,
        con_weight=1,
        theta=4,
        beta=0.56,
        rmax=6,
        C1=1,
        C2=0.8,
        C3=0.25,
        C4=0.25,
        noise_mu=2.3,
        noise_std=1.8,
        noise_tau=0.0054,
        delay_scale=0,
    ):
        # fiber_con = np.multiply(self.fiber_con, con_scale, out=None)
        fiber_con = self.fiber_con * con_weight

        if delay_scale.size == 1 and not np.isscalar(delay_scale):
            delay_scale = delay_scale[0]

        p = self.load_noise(
            tau=noise_tau, noise_mean=noise_mu * 100, noise_std=noise_std * 10
        )

        delay = self.delay * delay_scale if delay_scale else self.delay

        JR_network = JRNetwork_optim(
            N=self.nodes,
            W=fiber_con,
            dt=1 / self.source_sfreq,
            tau_exc1=tau_e1 / 100,
            tau_inh1=tau_i1 / 100,
            tau_exc2=tau_e2 / 100,
            tau_inh2=tau_i2 / 100,
            C1=C1,
            C2=C2,
            C3=C3,
            C4=C4,
            theta=theta,
            beta=beta,
            rmax=rmax,
            # theta=4, beta=0.54, rmax=5,
            delay=delay * 10,
            node_mode="double",
            band_w=band_w,
        )

        out = JR_network.simulate(tmax=self.simulate_times, P=p, sigma_p=22)
        return out.T  # trans to (node, simulate)

    def EEG_coregistration(self, jr_out):
        eeg_out = np.dot(self.fwd_sol, jr_out * self.amp_scale)
        if self.source_sfreq > self.eeg_sfreq:
            eeg_out = set_data_RFN(
                eeg_out, self.source_sfreq, self.eeg_sfreq, None, None
            )
        eeg_out = eeg_out - eeg_out.mean(axis=-1, keepdims=True)  # remove baseline
        eeg_out = eeg_out - eeg_out.mean(axis=0, keepdims=True)  # reference average
        return eeg_out
