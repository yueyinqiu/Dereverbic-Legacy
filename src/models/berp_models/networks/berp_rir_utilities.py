from statictorch import Tensor0d, Tensor1d
import torch
import scipy
from audio_processors.rir_acoustic_features import RirAcousticFeatures1d
from basic_utilities.static_class import StaticClass
from models.berp_models.networks.berp_temporal_envelope import BerpTemporalEnvelope


class BerpRirUtilities(StaticClass):
    @staticmethod
    def get_t_h(h: Tensor1d, fs = 16000) -> Tensor0d:
        h_cpu: torch.Tensor = h.to("cpu")

        temporal_env: BerpTemporalEnvelope = BerpTemporalEnvelope(dim=0, fc=20, fs=fs, mode="TAE")
        eh: torch.Tensor = temporal_env(h_cpu)

        Pks: torch.Tensor = torch.max(eh)
        I_t0: torch.Tensor = torch.argmax(eh)
        t0: torch.Tensor = I_t0 / fs

        riseside: torch.Tensor = eh[:I_t0]
        xT_Th: torch.Tensor = torch.arange(0, len(riseside)) / fs
        try:
            def objfunc(x, *params):
                return params[0] * torch.exp(6.9 * x / params[1])
            fit: list[float | torch.Tensor]
            fit, _ = scipy.optimize.curve_fit(
                objfunc,
                xT_Th,
                riseside,
                p0=[0.01, 0.01],
                bounds=([0.0, Pks], [0.01, t0]),
                method="trf",
            )
        except:
            fit = [0.0000, t0]  # Th does not exist
            # print("Th does not exist. Set to t0.")

        return Tensor0d(torch.as_tensor(fit[1]).to(h))

    @staticmethod
    def _polyval(p, x) -> torch.Tensor:
        p = torch.as_tensor(p)
        if p.ndim == 0 or (len(p) < 1):
            return torch.zeros_like(x)
        y: torch.Tensor = torch.zeros_like(x)
        p_i: torch.Tensor
        for p_i in p:
            y = y * x + p_i
        return y

    @staticmethod
    def get_t_t(h: Tensor1d, fs = 16000, fallback: bool = True) -> Tensor0d:
        h_cpu: torch.Tensor = h.to("cpu")

        # Actually it's just T60,
        # but there is a slight difference between 
        # the BERP's implemention here and 
        # our RirAcousticFeatures.get_reverberation_time
        # To ensure the result meet the SSIR's theory, we use this implemention in their model.

        x: torch.Tensor = torch.cumulative_trapezoid(h_cpu.flip(0) ** 2).double()
        x = x.flip(0)

        EDC: torch.Tensor = 10 * torch.log10(x)
        EDC = EDC - torch.max(EDC)
        I: torch.Tensor = torch.arange(1, len(EDC) + 1)

        xT1: torch.Tensor | float = torch.where(EDC <= 0)[0][0]
        if xT1 == []:
            xT1 = 1

        xT2: torch.Tensor | float = torch.where(EDC <= -20)[0][0]
        if xT2 == []:
            xT2 = torch.min(EDC)

        def linearfit(sta, end, EDC):
            I_xT: torch.Tensor = torch.arange(sta, end + 1)
            I_xT = I_xT.reshape(I_xT.numel(), 1)
            A: torch.Tensor = I_xT ** torch.arange(1, -1, -1.0).double()
            p: torch.Tensor = torch.linalg.inv(A.T @ A) @ (A.T @ EDC[I_xT])
            fittedline: torch.Tensor = BerpRirUtilities._polyval(p, I)
            return fittedline

        fittedline: torch.Tensor = linearfit(xT1, xT2, EDC)
        fittedline = fittedline - torch.max(fittedline)

        try:
            if torch.where(fittedline <= -18.2)[0].numel() == 0:
                xT2 = torch.where(EDC <= -10)[0][0]
                fittedline = linearfit(xT1, xT2, EDC)
                fittedline = fittedline - torch.max(fittedline)
                xT_60: torch.Tensor = 3.3 * torch.where(fittedline <= -18.2)[0][0]
            else:
                xT_60 = 3.3 * torch.where(fittedline <= -18.2)[0][0]
        except:
            if fallback:
                return Tensor0d(RirAcousticFeatures1d(h).reverberation_time(sample_rate=fs) * (60 / 30))
            raise ValueError("# T60 does not exist, the signal is not an RIR.")
        
        return Tensor0d((xT_60 / fs).to(h))