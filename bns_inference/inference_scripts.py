from sklearn.metrics import roc_curve, auc
import numpy as np


class DetectionAnalysis:
    def __init__(self, s_snr, y_true, y_score):
        """
        Initialize with SNR values, true labels, and scores for ROC computation.
        """
        self.s_snr_all = np.array(s_snr)  # all events
        self.y_true = np.array(y_true)
        self.y_score = np.array(y_score)

        # Extract only the signal events
        self.s_snr_signal = self.s_snr_all[self.y_true == 1]
        self.s_score_signal = self.y_score[self.y_true == 1]

    def compute_roc(self):
        """
        Compute ROC curve and AUC.
        Returns: fpr, tpr, roc_auc
        """
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_score)
        self.roc_auc = auc(self.fpr, self.tpr)
        return np.array(self.fpr), np.array(self.tpr), self.roc_auc

    def compute_efficiency(self, target_fpr=0.1, bin_step=2):
        """
        Compute detection efficiency vs SNR at a given target FPR.
        Automatically sets bins based on min/max of s_snr_signal.
        Returns: bin_centers, efficiency, errors
        """

        # Threshold corresponding to target FPR
        idx = np.argmin(np.abs(self.fpr - target_fpr))
        threshold = self.thresholds[idx]

        # Bin range based on signal SNR
        bin_start = np.floor(self.s_snr_signal.min())
        bin_stop = np.ceil(self.s_snr_signal.max()) + bin_step  # include last bin
        bins = np.arange(bin_start, bin_stop, bin_step)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        efficiencies = []
        errors = []
        for i in range(len(bins) - 1):
            in_bin = (self.s_snr_signal >= bins[i]) & (self.s_snr_signal < bins[i + 1])
            n_in_bin = np.sum(in_bin)
            if n_in_bin == 0:
                efficiencies.append(np.nan)
                errors.append(np.nan)
                continue
            detected = np.sum(self.s_score_signal[in_bin] > threshold)
            eff = detected / n_in_bin
            err = np.sqrt(eff * (1 - eff) / n_in_bin)

            efficiencies.append(eff)
            errors.append(err)

        return np.array(bin_centers), np.array(efficiencies), np.array(errors)
