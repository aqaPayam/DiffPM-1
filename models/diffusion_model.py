import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from models.schedule import cosine_beta_schedule
from models.unet import ImprovedDiffusionUNet1D


class DiffusionModel(nn.Module):
    def __init__(
        self,
        window_size: int,
        in_channels: int,
        time_emb_dim: int,
        base_channels: int,
        n_res_blocks: int,
        timesteps: int,
        s: float = 0.007,
        # new options
        use_v_prediction: bool = False,
        # default to DDPM; set sampler="ddim" to use DDIM
        sampler: str = "ddpm",                  # "ddpm" | "ddim"
        ddim_steps: Optional[int] = None,       # number of steps for DDIM (<= T). None -> T
        ddim_eta: float = 0.0,                  # 0 = deterministic DDIM
        normalize_per_feature: bool = False,    # enable z-score normalization buffers
        # NEW: optional external context (e.g. from LSTM)
        use_context: bool = False,
        context_dim: Optional[int] = None,
    ):
        """
        Args (original):
            window_size: length of each 1D window (W)
            in_channels: number of feature channels (D)
            time_emb_dim: embedding dim for time/cond
            base_channels, n_res_blocks: UNet config
            timesteps: total diffusion steps (T)
            s: offset in cosine schedule

        New:
            use_v_prediction: if True, the model predicts v instead of epsilon
            sampler: "ddpm" or "ddim"
            ddim_steps: number of inference steps for DDIM
            ddim_eta: stochasticity for DDIM (0 = deterministic)
            normalize_per_feature: if True, apply z-score per feature using buffers

            use_context: if True, the underlying UNet expects an additional
                         context vector per sample (e.g. from an LSTM over windows).
            context_dim: dimension of that external context vector. If None and
                         use_context=True, defaults to time_emb_dim.
        """
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.T = timesteps
        self.use_v_prediction = use_v_prediction
        self.sampler = sampler.lower()
        assert self.sampler in {"ddpm", "ddim"}, "sampler must be 'ddpm' or 'ddim'"
        self.ddim_steps = ddim_steps
        self.ddim_eta = float(ddim_eta)
        self.normalize_per_feature = normalize_per_feature

        # context-related flags (for LSTM-style conditioning)
        self.use_context = use_context
        self.context_dim = context_dim

        # noise schedule
        betas, alphas, alpha_bars = cosine_beta_schedule(timesteps, s)
        self.register_buffer('betas', betas)              # (T,)
        self.register_buffer('alphas', alphas)            # (T,)
        self.register_buffer('alpha_bars', alpha_bars)    # (T,)

        # posterior (improved DDPM variance)
        tilde_betas = betas.clone()
        tilde_betas[0] = betas[0]
        tilde_betas[1:] = ((1.0 - alpha_bars[:-1]) / (1.0 - alpha_bars[1:])) * betas[1:]
        self.register_buffer('tilde_betas', tilde_betas)  # (T,)

        # z-score normalization buffers (optional)
        if normalize_per_feature:
            self.register_buffer('feat_mean', torch.zeros(in_channels))
            self.register_buffer('feat_std', torch.ones(in_channels))
        else:
            self.feat_mean = None
            self.feat_std = None

        # diffusion model (1D U-Net)
        self.model = ImprovedDiffusionUNet1D(
            in_channels=in_channels,
            window_size=window_size,
            time_emb_dim=time_emb_dim,
            base_channels=base_channels,
            n_res_blocks=n_res_blocks,
            use_context=self.use_context,
            context_dim=self.context_dim,
        )

    # --------- normalization helpers ---------
    @torch.no_grad()
    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Provide per-feature z-score statistics.
        mean, std: shape (D,)
        """
        if not self.normalize_per_feature:
            raise ValueError("normalize_per_feature=False; enable it to set stats.")
        assert mean.shape == (self.in_channels,)
        assert std.shape == (self.in_channels,)
        # avoid zeros
        std = torch.where(std > 0, std, torch.ones_like(std))
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_per_feature:
            return x
        # x: (B, W, D)
        return (x - self.feat_mean.view(1, 1, -1)) / self.feat_std.view(1, 1, -1)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_per_feature:
            return x
        return x * self.feat_std.view(1, 1, -1) + self.feat_mean.view(1, 1, -1)

    # --------- training ---------
    def forward(
        self,
        x0: torch.Tensor,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss. Supports epsilon- or v-prediction.

        Args:
            x0:         (B, W, D) clean window
            start_idx:  (B,)
            series_len: (B,)
            context:    (B, C_ctx), optional external context per window
                        (e.g. LSTM hidden state). Only used if self.use_context=True.
        """
        assert x0.dim() == 3 and x0.size(2) == self.in_channels, \
            f'Expected input shape (B, W, {self.in_channels}), got {tuple(x0.shape)}'

        if self.use_context:
            assert context is not None, "use_context=True but no context was provided"
            assert context.dim() == 2 and context.size(0) == x0.size(0), \
                f"context must be (B, C_ctx), got {tuple(context.shape)}"

        x0 = self._normalize(x0)  # normalize before noising if enabled

        B = x0.size(0)
        device = x0.device

        # random timestep per example
        t = torch.randint(0, self.T, (B,), device=device)

        # sample noise
        eps = torch.randn_like(x0)

        # forward noising
        a_bar = self.alpha_bars[t].view(B, 1, 1)
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps

        # target depends on prediction parametrization
        if self.use_v_prediction:
            # v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0
            target = torch.sqrt(a_bar) * eps - torch.sqrt(1 - a_bar) * x0
        else:
            target = eps

        pred = self.model(x_t, t, start_idx, series_len, context=context)
        return F.mse_loss(pred, target)

    # --------- utilities ---------
    def _predict_eps_from_model(
        self,
        x_t: torch.Tensor,
        t: torch.LongTensor,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns epsilon prediction regardless of model parametrization.

        Args:
            x_t:        (B, W, D)
            t:          (B,)
            start_idx:  (B,)
            series_len: (B,)
            context:    (B, C_ctx), optional external context per window
        """
        model_out = self.model(x_t, t, start_idx, series_len, context=context)
        if self.use_v_prediction:
            # eps = sqrt(a_bar)*v + sqrt(1 - a_bar)*x_t
            a_bar = self.alpha_bars[t].view(-1, 1, 1)  # (B,1,1)
            eps = torch.sqrt(a_bar) * model_out + torch.sqrt(1 - a_bar) * x_t
            return eps
        else:
            return model_out  # eps-pred

    # --------- DDPM ancestral sampler ---------
    @torch.no_grad()
    def _sample_ddpm(
        self,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
        device: Optional[torch.device] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample windows using DDPM ancestral sampling.

        Args:
            start_idx:  (B,)
            series_len: (B,)
            context:    (B, C_ctx), optional; if provided, reused across timesteps
        """
        if device is None:
            device = self.betas.device

        start_idx = start_idx.to(device)
        series_len = series_len.to(device)
        B = start_idx.size(0)

        if self.use_context and context is not None:
            context = context.to(device)
            assert context.dim() == 2 and context.size(0) == B, \
                f"context must be (B, C_ctx), got {tuple(context.shape)}"
        elif self.use_context and context is None:
            raise ValueError("use_context=True but no context was provided to _sample_ddpm")

        x = torch.randn(B, self.window_size, self.in_channels, device=device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            eps_pred = self._predict_eps_from_model(
                x, t_tensor, start_idx, series_len, context=context
            )

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            a_bar_t = self.alpha_bars[t]

            # mean of q(x_{t-1} | x_t, x0) expressed via eps_pred
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - a_bar_t)
            x_prev_mean = coef1 * (x - coef2 * eps_pred)

            if t > 0:
                # improved posterior variance
                sigma_t = torch.sqrt(self.tilde_betas[t])
                noise = torch.randn_like(x)
                x = x_prev_mean + sigma_t * noise
            else:
                x = x_prev_mean

        # denormalize if needed
        x = self._denormalize(x)
        return x

    # --------- DDIM sampler (can use fewer steps, eta controls noise) ---------
    @torch.no_grad()
    def _sample_ddim(
        self,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
        device: Optional[torch.device] = None,
        ddim_steps: Optional[int] = None,
        eta: Optional[float] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample windows using DDIM.

        Args:
            start_idx:  (B,)
            series_len: (B,)
            context:    (B, C_ctx), optional; if provided, reused across timesteps
        """
        if device is None:
            device = self.betas.device
        if ddim_steps is None:
            ddim_steps = self.ddim_steps or self.T
        if eta is None:
            eta = self.ddim_eta

        start_idx = start_idx.to(device)
        series_len = series_len.to(device)
        B = start_idx.size(0)

        if self.use_context and context is not None:
            context = context.to(device)
            assert context.dim() == 2 and context.size(0) == B, \
                f"context must be (B, C_ctx), got {tuple(context.shape)}"
        elif self.use_context and context is None:
            raise ValueError("use_context=True but no context was provided to _sample_ddim")

        # choose timesteps (uniform spacing)
        if ddim_steps > self.T:
            raise ValueError("ddim_steps must be <= total timesteps T")
        step_indices = torch.linspace(self.T - 1, 0, steps=ddim_steps, device=device).long()

        x = torch.randn(B, self.window_size, self.in_channels, device=device)

        for i, t in enumerate(step_indices):
            t_int = int(t.item())
            t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)

            a_bar_t = self.alpha_bars[t_int]
            eps_pred = self._predict_eps_from_model(
                x, t_tensor, start_idx, series_len, context=context
            )

            # predict x0
            x0_pred = (x - torch.sqrt(1.0 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)
            if i == len(step_indices) - 1:
                x = x0_pred
                break

            t_prev = int(step_indices[i + 1].item())
            a_bar_prev = self.alpha_bars[t_prev]

            # variance for DDIM with eta
            # sigma_t controls residual noise injected
            sigma_t = eta * torch.sqrt(
                (1.0 - a_bar_prev) / (1.0 - a_bar_t) * (1.0 - a_bar_t / a_bar_prev)
            )

            # direction preserving term
            dir_xt = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t**2, min=0.0)) * eps_pred
            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = torch.sqrt(a_bar_prev) * x0_pred + dir_xt + sigma_t * noise

        x = self._denormalize(x)
        return x

    # --------- public window sampler ---------
    @torch.no_grad()
    def sample(
        self,
        start_idx: torch.LongTensor,
        series_len: torch.LongTensor,
        device: Optional[torch.device] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples by reversing the diffusion process for one window (B, W, D).
        Uses DDPM (posterior variance) or DDIM depending on `self.sampler`.

        Args:
            start_idx:  (B,)
            series_len: (B,)
            context:    (B, C_ctx), optional external context per window.
                        For LSTM-based conditioning, this would typically be
                        derived from previous windows and kept fixed while
                        sampling the current window.
        """
        if self.sampler == "ddpm":
            return self._sample_ddpm(start_idx, series_len, device, context=context)
        else:
            return self._sample_ddim(start_idx, series_len, device, context=context)

    # --------- full-series sampler with overlap ---------
    @torch.no_grad()
    def sample_total(
        self,
        start_idx: torch.LongTensor,   # (B,)
        end_idx: torch.LongTensor,     # (B,)
        shift: int,
        min_value: float,
        max_value: float,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Sample a full series from start to end by sliding and averaging overlapping windows,
        using batched sampling instead of sequential per-window loops.

        NOTE: this version does NOT make use of LSTM-style context across windows;
        all windows are sampled independently (as in the original design). For
        sequence-dependent conditioning, a separate sequential sampler would be
        more appropriate.

        Args:
            start_idx: (B,) tensor of start positions
            end_idx:   (B,) tensor of end positions
            shift:     int stride (1 <= shift < window_size)
            min_value: float minimum valid value
            max_value: float maximum valid value
            device:    torch device (defaults to model's device)

        Returns:
            Tensor of shape (B, L, D) where L = end_idx - start_idx + 1
        """
        if device is None:
            device = self.betas.device

        # Move to device
        start_idx = start_idx.to(device)    # (B,)
        end_idx   = end_idx.to(device)      # (B,)
        B = start_idx.size(0)

        # Ensure all series lengths L are equal across batch
        lengths = end_idx - start_idx + 1   # (B,)
        if not torch.all(lengths == lengths[0]):
            raise ValueError("All series lengths must be equal across batch")
        L = int(lengths[0].item())          # scalar

        # Validate shift
        if not (1 <= shift < self.window_size):
            raise ValueError(f"shift must be between 1 and {self.window_size-1}")

        # Compute window start positions (based on first sample)
        s0 = int(start_idx[0].item())
        e0 = s0 + L - 1
        positions = list(range(s0, e0 - self.window_size + 2, shift))
        num_windows = len(positions)

        # Expand positions for the whole batch
        ws_tensor = torch.tensor(positions, device=device, dtype=torch.long)  # (num_windows,)
        ws_tensor = ws_tensor.unsqueeze(0).expand(B, num_windows).reshape(-1) # (B*num_windows,)

        series_len_tens = torch.full((B*num_windows,), L, dtype=torch.long, device=device)

        # Run all windows in one batched call (independent, no context)
        x_win_batch = self.sample(ws_tensor, series_len_tens, device=device)
        # shape: (B*num_windows, W, D)

        # Reshape back: (B, num_windows, W, D)
        x_win_batch = x_win_batch.view(B, num_windows, self.window_size, self.in_channels)

        # Prepare accumulators
        sum_series = torch.zeros(B, L, self.in_channels, device=device)
        count      = torch.zeros(B, L, self.in_channels, device=device)

        # Iterate over window positions (small loop, just accumulation)
        for j, ws in enumerate(positions):
            x_win = x_win_batch[:, j, :, :]  # (B, W, D)

            # mask out-of-range entries
            mask = (x_win >= min_value) & (x_win <= max_value)
            mask_f = mask.float()

            offset = ws - s0  # scalar offset into [0, L-window_size]
            sum_series[:, offset:offset+self.window_size, :] += x_win * mask_f
            count[:,      offset:offset+self.window_size, :] += mask_f

        # Compute final averaged series, safe divide
        avg = torch.where(count > 0, sum_series / count, torch.zeros_like(sum_series))
        # avg: (B, L, D)
        return avg
