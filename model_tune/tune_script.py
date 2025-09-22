import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction, UniformComovingVolume
from torch.distributions import Uniform
from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw.gw import get_ifo_geometry, compute_observed_strain
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pathlib import Path
from ml4gw.transforms import SpectralDensity
import h5py
from ml4gw.gw import compute_ifo_snr, compute_network_snr
from ml4gw.gw import reweight_snrs
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import Whiten
from ml4gw.nn.resnet import ResNet1D

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import numpy as np
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import RayLightningEnvironment, RayTrainReportCallback, RayDDPStrategy, prepare_trainer
from ray.train.torch import TorchTrainer
import ray

data_dir = Path("./data")
background_dir = data_dir / "background_data"

param_dict = {
    "chirp_mass": Uniform(1.17, 2.2),
    "mass_ratio": Uniform(0.6, 1.0),
    "chi1": Uniform(-0.05, 0.05),
    "chi2": Uniform(-0.05, 0.05),
    "distance": UniformComovingVolume(10,500,distance_type='luminosity_distance'),
    "phic": DeltaFunction(0),
    "inclination": Sine(),
}

# MLP definition
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
from ml4gw import augmentations, distributions, gw, transforms, waveforms
from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset
from ml4gw.utils.slicing import sample_kernels
import torch
from lightning import pytorch as pl
import torchmetrics
from torchmetrics.classification import BinaryAUROC

from typing import Callable, Dict, List


class Ml4gwDetectionModel(pl.LightningModule):
    """
    Model with methods for generating waveforms and
    performing our preprocessing augmentations in
    real-time on the GPU. Also loads training background
    in chunks from disk, then samples batches from chunks.
    """

    def __init__(
        self,
        architecture_a: torch.nn.Module,
        architecture_b: torch.nn.Module,
        architecture_c: torch.nn.Module,
        architecture_d: torch.nn.Module,
        metric: torchmetrics.Metric,
        ifos: List[str] = ["H1", "L1"],
        kernel_length: float = 1.5, # this should the aframe window ()
        # PSD/whitening args
        fduration: float = 2,
        psd_length: float = 16, 
        sample_rate: float = 2048,
        fftlength: float = 2, # would we change this for BNS
        highpass: float = 32,
        # Dataloading args
        chunk_length: float = 128,  # we'll talk about chunks in a second
        reads_per_chunk: int = 40,
        learning_rate: float = 0.005,
        batch_size: int = 256,
        # Waveform generation args
        waveform_prob: float = 0.5,
        approximant: Callable = waveforms.cbc.IMRPhenomD,
        param_dict: Dict[str, torch.distributions.Distribution] = param_dict,
        waveform_duration: float = 60,
        f_min: float = 20,
        f_max: float = None,
        f_ref: float = 20,
        # Augmentation args
        inversion_prob: float = 0.5,
        reversal_prob: float = 0.5,
        min_snr: float = 30,
        max_snr: float = 100,
        snr_dist: str = "powerlaw",
        # Downsampling the injected signal
        variable_rate: bool = False,
        # Validation dataset
        val_filename: str = "validation_dataset.hdf5",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["architecture_a", "architecture_b", "architecture_c", "architecture_d", "metric", "approximant"]
        )
    
        self.nn_a = architecture_a
        self.nn_b = architecture_b
        self.nn_c = architecture_c
        self.nn_d = architecture_d

        self.metric = metric
        self.variable_rate = variable_rate

        self.inverter = augmentations.SignalInverter(prob=inversion_prob)
        self.reverser = augmentations.SignalReverser(prob=reversal_prob)

        # real-time transformations defined with torch Modules
        self.spectral_density = transforms.SpectralDensity(
            sample_rate, fftlength, average="median", fast=False
        )
        self.whitener = transforms.Whiten(
            fduration, sample_rate, highpass=highpass
        )

        # get some geometry information about
        # the interferometers we're going to project to
        detector_tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("detector_tensors", detector_tensors)
        self.register_buffer("detector_vertices", vertices)

        # define some sky parameter distributions
        self.param_dict = param_dict
        self.dec = distributions.Cosine()
        self.psi = torch.distributions.Uniform(0, torch.pi)
        self.phi = torch.distributions.Uniform(
            -torch.pi, torch.pi
        )  # relative RAs of detector and source
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant=approximant(),
            sample_rate=sample_rate,
            duration=waveform_duration,
            f_min=f_min,
            f_ref=f_ref,
            right_pad=0.5,
        ).to(self.device)

        # rather than sample distances, we'll sample target SNRs.
        # This way we can ensure we train our network on
        # signals that are more detectable. We'll use a distribution
        # that looks roughly like the natural sampled SNR distribution
        # self.snr = distributions.PowerLaw(min_snr, max_snr, -3)

        # define SNR distribution
        if snr_dist.lower() == "uniform":
            self.snr = torch.distributions.Uniform(min_snr, max_snr)
        elif snr_dist.lower() == "powerlaw":
            self.snr = distributions.PowerLaw(min_snr, max_snr, -3)
        else:
            raise ValueError(f"Unknown snr_dist {snr_dist}, must be 'uniform' or 'powerlaw'")

        # up front let's define some properties in units of samples
        # Note the different usage of window_size from earlier
        self.kernel_size = int(kernel_length * sample_rate)
        self.window_size = self.kernel_size + int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)

        # validation dataset file
        self.val_filename = val_filename

    def forward(self, X_a, X_b, X_c):
        ts_a = self.nn_a(X_a)
        ts_b = self.nn_b(X_b)
        ts_c = self.nn_c(X_c)
        ts_d = torch.cat((ts_a, ts_b, ts_c), dim=1)
        return self.nn_d(ts_d)

    def training_step(self, batch):
        X_a, X_b, X_c, y = batch
        y_hat = self(X_a, X_b, X_c)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        X_a, X_b, X_c, y = batch
        y_hat = self(X_a, X_b, X_c)
        self.metric.update(y_hat, y)
        self.log("valid_auroc", self.metric, on_epoch=True, prog_bar=True)
        return {"valid_auroc": self.metric}

    def configure_optimizers(self):
        # parameters = self.nn.parameters()
        parameters = list(self.nn_a.parameters()) + list(self.nn_b.parameters()) + list(self.nn_c.parameters()) + list(self.nn_d.parameters())
        optimizer = torch.optim.AdamW(parameters, self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            pct_start=0.1,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = dict(scheduler=scheduler, interval="step")
        return dict(optimizer=optimizer, lr_scheduler=scheduler_config)

    def configure_callbacks(self):
        chkpt = pl.callbacks.ModelCheckpoint(monitor="valid_auroc", save_top_k=10, mode="max")
        return [chkpt]

    def generate_waveforms(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        rvs = torch.rand(size=(batch_size,))
        mask = rvs < self.hparams.waveform_prob
        num_injections = mask.sum().item()

        params = {
            k: v.sample((num_injections,)).to(device)
            for k, v in self.param_dict.items()
        }

        params["s1z"], params["s2z"] = (
            params["chi1"], params["chi2"]
        )
        params["mass_1"], params["mass_2"] = waveforms.conversion.chirp_mass_and_mass_ratio_to_components(
            params["chirp_mass"], params["mass_ratio"]
        )

        hc, hp = self.waveform_generator(**params)
        return hc, hp, mask

    def project_waveforms(
        self, hc: torch.Tensor, hp: torch.Tensor
    ) -> torch.Tensor:
        # sample sky parameters
        N = len(hc)
        dec = self.dec.sample((N,)).to(hc)
        psi = self.psi.sample((N,)).to(hc)
        phi = self.phi.sample((N,)).to(hc)

        # project to interferometer response
        return gw.compute_observed_strain(
            dec=dec,
            psi=psi,
            phi=phi,
            detector_tensors=self.detector_tensors,
            detector_vertices=self.detector_vertices,
            sample_rate=self.hparams.sample_rate,
            cross=hc,
            plus=hp,
        )

    def rescale_snrs(
        self, responses: torch.Tensor, psd: torch.Tensor
    ) -> torch.Tensor:
        # make sure everything has the same number of frequency bins
        num_freqs = int(responses.size(-1) // 2) + 1
        if psd.size(-1) != num_freqs:
            psd = torch.nn.functional.interpolate(
                psd, size=(num_freqs,), mode="linear"
            )
        N = len(responses)
        target_snrs = self.snr.sample((N,)).to(responses.device)
        return gw.reweight_snrs(
            responses=responses.double(),
            target_snrs=target_snrs,
            psd=psd,
            sample_rate=self.hparams.sample_rate,
            highpass=self.hparams.highpass,
        )

    def sample_waveforms(self, responses: torch.Tensor) -> torch.Tensor:
        pad = int((self.hparams.fduration / 2) * self.hparams.sample_rate)
        responses = torch.nn.functional.pad(responses, [pad, pad])
        return responses[-int(self.window_size * self.hparams.sample_rate):]

        # slice off random views of each waveform to inject in arbitrary positions
        #responses = responses[:, :, -self.window_size :]

        # pad so that at least half the kernel always contains signals
        #pad = [0, int(self.window_size // 2)]
        # pad = [0,int(0.5*self.hparams.sample_rate)] # padding half a second to the right, waveform generator had the merger 0.5s to the left of the edge
        #responses = torch.nn.functional.pad(responses, pad)
        #return sample_kernels(responses, self.window_size, coincident=True)

    def build_variable_indices(self, sr=2048, schedule=None, device=None):
        if schedule is None:
            schedule = torch.tensor([[0, 40, 256],
                                    [40, 58, 512],
                                    [58, 60, 2048]], dtype=torch.int, device=device)

        idx = torch.tensor([], dtype=torch.long, device=device)
        for s in schedule:
            if idx.size()[0] == 0:
                start = int(s[0] * sr)
            else:
                start = int(idx[-1]) + int(idx[-1] - idx[-2])
            stop = int(start + (s[1] - s[0]) * sr)
            step = int(sr // s[2])
            idx = torch.cat((idx, torch.arange(start, stop, step, dtype=torch.int, device=device)))
        return idx
    
    def split_by_schedule(self, signal, schedule=None, device=None):
        if schedule is None:
            schedule = torch.tensor([[0, 40, 256],
                                     [40, 58, 512],
                                     [58, 60, 2048]], device=device)
            
        split = schedule[:,2].unsqueeze(-1) * (schedule[:,1].unsqueeze(-1) - schedule[:,0].unsqueeze(-1))
        d1 = torch.split(signal, split.squeeze().tolist(), dim=-1)
        return d1

    @torch.no_grad()
    def augment(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # break off "background" from target kernel and compute its PSD
        # (in double precision since our scale is so small)
        background, X = torch.split(
            X, [self.psd_size, self.window_size], dim=-1
        )
        psd = self.spectral_density(background.double())

        # Generate at most batch_size signals from our parameter distributions
        # Keep a mask that indicates which rows to inject in
        batch_size = X.size(0)
        hc, hp, mask = self.generate_waveforms(batch_size)
        hc, hp, mask = hc, hp, mask

        # Augment with inversion and reversal
        X = self.inverter(X)
        X = self.reverser(X)

        # sample sky parameters and project to responses, then
        # rescale the response according to a randomly sampled SNR
        responses = self.project_waveforms(hc, hp)
        responses = self.rescale_snrs(responses, psd[mask])

        # randomly slice out a window of the waveform, add it
        # to our background, then whiten everything
        responses = self.sample_waveforms(responses)
        # print(responses.shape, X.shape)

        X[mask] += responses.float()
        X = self.whitener(X, psd)
        # X1 = X.clone()

        # this is where I implement the downsampling for the dataset
        if self.variable_rate:
            indices = self.build_variable_indices(
                sr=self.hparams.sample_rate,
                device=X.device)
            
            X = X.index_select(dim=-1, index=indices)
            # X2 = X.clone()

            #after downsampling, splitting it into different time segment series
            X = self.split_by_schedule(X) 
            X_a, X_b, X_c = X[0], X[1], X[2]

        # create labels, marking 1s where we injected
        y = torch.zeros((batch_size, 1), device=X_a.device)
        y[mask] = 1
        return X_a, X_b, X_c, y

    def on_after_batch_transfer(self, batch, _):
        # this is a parent method that lightning calls
        # between when the batch gets moved to GPU and
        # when it gets passed to the training_step.
        # Apply our augmentations here
        if self.trainer.training:
            batch = self.augment(batch)
        return batch

    def train_dataloader(self):
        # Because our entire training dataset is generated
        # on the fly, the traditional idea of an "epoch"
        # meaning one pass through the training set doesn't
        # apply here. Instead, we have to set the number
        # of batches per epoch ourselves, which really
        # just amounts to deciding how often we want
        # to run over the validation dataset.
        samples_per_epoch = 3000
        batches_per_epoch = (
            int((samples_per_epoch - 1) // self.hparams.batch_size) + 1
        )
        batches_per_chunk = int(batches_per_epoch // 10)
        chunks_per_epoch = int(batches_per_epoch // batches_per_chunk) + 1

        # Hdf5TimeSeries dataset samples batches from disk.
        # In this instance, we'll make our batches really large so that
        # we can treat them as chunks to sample training batches from
        fnames = list(background_dir.iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=self.hparams.ifos,
            kernel_size=int(
                self.hparams.chunk_length * self.hparams.sample_rate
            ),
            batch_size=self.hparams.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch,
            coincident=False,
        )

        # sample batches to pass to our NN from the chunks loaded from disk
        return ChunkedTimeSeriesDataset(
            dataset,
            kernel_size=self.window_size + self.psd_size,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
            coincident=False,
        )

    def val_dataloader(self):
        with h5py.File(data_dir / self.val_filename, "r") as f:
            X = torch.Tensor(f["X"][:])
            y = torch.Tensor(f["y"][:])

        if self.variable_rate:
            indices = self.build_variable_indices(
                sr=self.hparams.sample_rate,
                device=X.device)
            
            X = X.index_select(dim=-1, index=indices)

            X = self.split_by_schedule(X) 
            X_a, X_b, X_c = X[0], X[1], X[2]

        dataset = torch.utils.data.TensorDataset(X_a, X_b, X_c, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 4,
            shuffle=False,
            pin_memory=True,
        )

def train_tune(config):
    kernel_size_a = config["kernel_size_a"]
    kernel_size_b = config["kernel_size_b"]
    kernel_size_c = config["kernel_size_c"]
    learning_rate = config["learning_rate"]

    architecture_a = ResNet1D(
        in_channels=2, layers=[2, 2], classes=1, kernel_size=kernel_size_a
    ).to(device)
    architecture_b = ResNet1D(
        in_channels=2, layers=[2, 2], classes=1, kernel_size=kernel_size_b
    ).to(device)
    architecture_c = ResNet1D(
        in_channels=2, layers=[2, 2], classes=1, kernel_size=kernel_size_c
    ).to(device)
    architecture_d = MLP(input_size=3, hidden_size=64, output_size=1).to(device)
    metric = BinaryAUROC(max_fpr=1e-3)

    model = Ml4gwDetectionModel(
        kernel_length=60,
        batch_size=128,
        snr_dist="uniform",
        min_snr=8,
        max_snr=100,
        architecture_a=architecture_a,
        architecture_b=architecture_b,
        architecture_c=architecture_c,
        architecture_d=architecture_d,
        metric=metric,
        variable_rate=True,
        psd_length=20,
        learning_rate=learning_rate,
        val_filename="validation_dataset_1.hdf5",
    )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPSTrategy(find_unused_parameters=True),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
        precision="16-mixed",
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model=model)

def tune_with_asha(ray_trainer, scheduler, num_samples=10):
        tuner = tune.tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="valid_auroc",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        
        return tuner.fit()

if __name__ == "__main__":
    import torch
    import socket

    print("CUDA present {} on {}".format(torch.cuda.is_available(), socket.gethostname()))
    ray.init()

    search_space = {
    "kernel_size_a": tune.choice([3, 5, 7]),
    "kernel_size_b": tune.choice([3, 5, 7]),
    "kernel_size_c": tune.choice([3, 5, 7]),
    "learning_rate": tune.loguniform(5e-4, 5e-3),
    }
    
    num_epochs = 10

    num_samples = 20

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=5, reduction_factor=2)

    scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, 
    resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path="/home/bhgupta/orcd/scratch/mlgw/data/logs/ray_tune_results",
        name="trial_1",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="valid_auroc",
            checkpoint_score_order="max",
        ),
    )

    ray_trainer = TorchTrainer(
        train_tune, 
        scaling_config=scaling_config,
        run_config=run_config,
    )
    results = tune_with_asha(ray_trainer, scheduler, num_samples=num_samples)
    print("Best hyperparameters found were: ", results.get_best_result().config)