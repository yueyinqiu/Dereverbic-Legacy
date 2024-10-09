import torch as _torch

class RicModule(_torch.nn.Module):
    def __init__(self, 
                 input_clean_length: int, 
                 input_reverb_length: int, 
                 output_rir_length: int):
        super().__init__()
        self.network: _torch.nn.Sequential = _torch.nn.Sequential(
            _torch.nn.Conv1d(6, 16, 5, padding=2),
            _torch.nn.ReLU(),
            _torch.nn.Conv1d(16, 16, 5, padding=2), 
            _torch.nn.ReLU(),
            _torch.nn.Conv1d(16, 8, 5, padding=2), 
            _torch.nn.ReLU(),
            _torch.nn.Conv1d(8, 1, 5, padding=2), 
            _torch.nn.ReLU())

    def forward(self, clean_batch: _torch.Tensor, reverb_batch: _torch.Tensor):
        clean_batch_fft: _torch.Tensor = _torch.fft.fft(clean_batch)
        clean_batch_fft_real: _torch.Tensor = clean_batch_fft.real
        clean_batch_fft_imag: _torch.Tensor = clean_batch_fft.imag

        reverberant_batch_fft: _torch.Tensor = _torch.fft.fft(reverb_batch)
        reverberant_batch_fft_real: _torch.Tensor = reverberant_batch_fft.real
        reverberant_batch_fft_imag: _torch.Tensor = reverberant_batch_fft.imag

        network_input: _torch.Tensor = _torch.stack(
            [clean_batch, clean_batch_fft_real, clean_batch_fft_imag,
             reverb_batch, reverberant_batch_fft_real, reverberant_batch_fft_imag], dim=1)

        result: _torch.Tensor = self.network(network_input)
        return result.squeeze(1)
        
