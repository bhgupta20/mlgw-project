import torch
import h5py
from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction, UniformComovingVolume
from torch.distributions import Uniform
from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw.gw import get_ifo_geometry, compute_observed_strain
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pathlib import Path
from ml4gw.transforms import SpectralDensity
from ml4gw.gw import compute_ifo_snr, compute_network_snr
from ml4gw.gw import reweight_snrs
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import Whiten

data_dir = Path("./data")
background_dir = data_dir / "background_data_1"
ifos = ["H1", "L1"]

total_samples = 50000
batch_size = 50
n_batches = total_samples // batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"

waveform_duration = 60
sample_rate = 2048
f_min = 20
f_max = 1024
f_ref = 20

num_samples = int(waveform_duration * sample_rate)
num_freqs = num_samples // 2 + 1
frequencies = torch.linspace(0, sample_rate/2, num_freqs).to(device)
freq_mask = (frequencies >= f_min) & (frequencies < f_max)

param_dict = {
    "chirp_mass": Uniform(1.17, 2.2),
    "mass_ratio": Uniform(0.6, 1.0),
    "chi1": Uniform(-0.05, 0.05),
    "chi2": Uniform(-0.05, 0.05),
    "distance": UniformComovingVolume(10,500,distance_type='luminosity_distance'),
    "phic": DeltaFunction(0),
    "inclination": Sine(),
}

approximant = IMRPhenomD().to(device)
waveform_generator = TimeDomainCBCWaveformGenerator(
    approximant=approximant,
    sample_rate=sample_rate,
    f_min=f_min,
    duration=waveform_duration,
    right_pad=0.5,
    f_ref=f_ref,
).to(device)

tensors, vertices = get_ifo_geometry(*ifos)
tensors, vertices = tensors.to(device), vertices.to(device)

psd_length = 20
fduration = 2

whiten = Whiten(
    fduration=fduration, sample_rate=sample_rate, highpass=f_min
).to(device)

def generate_batch(batch_size, device=device):

    params = {k: v.sample((batch_size,)).to(device) for k, v in param_dict.items()}
    params["mass_1"], params["mass_2"] = chirp_mass_and_mass_ratio_to_components(
        params["chirp_mass"], params["mass_ratio"]
    )
    params["s1z"], params["s2z"] = params["chi1"], params["chi2"]

    hc, hp = waveform_generator(**params)

    dec = Sine()
    psi = Uniform(0, torch.pi)
    phi = Uniform(-torch.pi, torch.pi)
    responses = compute_observed_strain(
        dec=dec.sample((batch_size,)).to(device),
        psi=psi.sample((batch_size,)).to(device),
        phi=phi.sample((batch_size,)).to(device),
        detector_tensors=tensors,
        detector_vertices=vertices,
        sample_rate=sample_rate,
        cross=hc,
        plus=hp
    )

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=2,
        overlap=None,
        average="median"
    ).to(device)
    
    fnames = [list(background_dir.iterdir())[0]]
    with h5py.File(fnames[0], "r") as f:
        background = [torch.Tensor(f[ifo][:]) for ifo in ifos]
        background = torch.stack(background).to(device)
    
    psd = spectral_density(background.double())

    if psd.shape[-1] != num_freqs:
        # Adding dummy dimensions for consistency
        while psd.ndim < 3:
            psd = psd[None]
        psd = torch.nn.functional.interpolate(
            psd, size=(num_freqs,), mode="linear"
        )

    target_snrs = Uniform(8, 100).sample((batch_size,)).to(device)

    responses = reweight_snrs(
        responses=responses,
        target_snrs=target_snrs,
        psd=psd,
        sample_rate=sample_rate,
        highpass=f_min
    )

    psd_size = int(psd_length * sample_rate)
    kernel_size = int(waveform_duration * sample_rate)
    window_length = psd_length + fduration + waveform_duration

    fnames = list(background_dir.iterdir())
    dataloader = Hdf5TimeSeriesDataset(
    fnames=fnames,
    channels=ifos,
    kernel_size=int(window_length * sample_rate),
    batch_size=2
    * batch_size,
    batches_per_epoch=1,
    coincident=False,
    )

    background_samples = [x for x in dataloader][0].to(device)
    psd = spectral_density(background_samples[..., :psd_size].double())
    kernel = background_samples[..., psd_size:]

    pad = int(fduration / 2 * sample_rate)
    injected = kernel.detach().clone()
    injected[::2, :, pad:-pad] += responses[..., -kernel_size:]
    whitened = whiten(injected, psd)

    y = torch.zeros(len(injected))
    y[::2] = 1
    snr = torch.zeros(len(injected))
    snr[::2] = target_snrs

    return whitened.cpu(), y.cpu(), snr.cpu()

def main():

    out_file = "/ceph/submit/data/user/b/bhgupta/test_dataset_5.hdf5"
    # out_file = data_dir / "test_dataset_5.hdf5"
    with h5py.File(out_file, "w") as f:
    
        dset_X = f.create_dataset("X", shape=(2*total_samples, 2, 122880), dtype='float32')
        dset_y = f.create_dataset("y", shape=(2*total_samples,), dtype='int64')
        dset_snr = f.create_dataset("snr", shape=(2*total_samples,), dtype='float32')

        idx = 0
        for i in range(n_batches):
            print(f"Generating batch {i+1}/{n_batches}")
            X, y, snr = generate_batch(batch_size, device=device)

            X = X.cpu().numpy()
            y = y.cpu().numpy()
            snr = snr.cpu().numpy()

            dset_X[idx:idx+2*batch_size] = X
            dset_y[idx:idx+2*batch_size] = y
            dset_snr[idx:idx+2*batch_size] = snr
            idx += 2*batch_size

if __name__ == "__main__":
    main()