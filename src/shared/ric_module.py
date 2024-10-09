import torch

class RicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv1d(6, 16, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, 5, padding=2), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 8, 5, padding=2), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 1, 5, padding=2), 
            torch.nn.ReLU())

    def forward(self, clean_batch: torch.Tensor, reverberant_batch: torch.Tensor):
        clean_batch_fft: torch.Tensor = torch.fft.fft(clean_batch)
        clean_batch_fft_real: torch.Tensor = clean_batch_fft.real
        clean_batch_fft_imag: torch.Tensor = clean_batch_fft.imag

        reverberant_batch_fft: torch.Tensor = torch.fft.fft(reverberant_batch)
        reverberant_batch_fft_real: torch.Tensor = reverberant_batch_fft.real
        reverberant_batch_fft_imag: torch.Tensor = reverberant_batch_fft.imag

        network_input: torch.Tensor = torch.stack(
            [clean_batch, clean_batch_fft_real, clean_batch_fft_imag,
             reverberant_batch, reverberant_batch_fft_real, reverberant_batch_fft_imag], dim=1)

        return self.network(network_input)

        
