import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for MedTrust DenseNet-121.

    Generates a class-specific activation map from
    convolutional feature activations and their gradients.

    Expected model output:
        [B, num_classes] raw logits
    """

    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None

        self._register_hooks()

    # =====================================================
    # HOOKS
    # =====================================================

    def _register_hooks(self):

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        """
        Save activations and attach a gradient hook directly
        to the activation tensor.

        We intentionally avoid register_full_backward_hook()
        because torchvision DenseNet applies an in-place
        ReLU after features, which can conflict with module
        backward hooks.
        """

        self.activations = output

        if output.requires_grad:

            output.register_hook(
                self._save_gradient
            )

    def _save_gradient(
        self,
        gradient,
    ):
        self.gradients = gradient.detach()

    # =====================================================
    # GENERATE GRAD-CAM
    # =====================================================

    def generate(
        self,
        input_tensor,
        class_index,
    ):
        """
        Generate Grad-CAM for one disease class.

        Parameters
        ----------
        input_tensor:
            Tensor with shape [1, 3, H, W].

        class_index:
            Disease class index whose raw model logit
            should be explained.

        Returns
        -------
        cam:
            Tensor [H, W], normalized to [0, 1].

        logits:
            Raw model logits.
        """

        if input_tensor.ndim != 4:
            raise ValueError(
                "input_tensor must have shape "
                "[B, C, H, W]."
            )

        if input_tensor.size(0) != 1:
            raise ValueError(
                "GradCAM.generate currently expects "
                "batch size 1."
            )

        self.activations = None
        self.gradients = None

        # Grad-CAM requires autograd.
        # Do NOT use inference_mode() or no_grad().
        with torch.enable_grad():

            self.model.zero_grad(
                set_to_none=True
            )

            logits = self.model(
                input_tensor
            )

            if logits.ndim != 2:
                raise RuntimeError(
                    "Expected model output shape "
                    "[B, num_classes]."
                )

            if not (
                0 <= class_index < logits.size(1)
            ):
                raise IndexError(
                    f"class_index={class_index} "
                    f"is outside valid range "
                    f"0-{logits.size(1) - 1}."
                )

            target_logit = logits[
                0,
                class_index,
            ]

            target_logit.backward()

        # -------------------------------------------------
        # Sanity checks
        # -------------------------------------------------

        if self.activations is None:
            raise RuntimeError(
                "Forward activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Backward gradients were not captured."
            )

        activations = (
            self.activations.detach()
        )

        gradients = self.gradients

        # -------------------------------------------------
        # Grad-CAM weights
        #
        # gradients:
        # [1, C, Hf, Wf]
        #
        # weights:
        # [1, C, 1, 1]
        # -------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # -------------------------------------------------
        # Weighted feature combination
        # -------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Keep positive contributions.
        cam = F.relu(
            cam
        )

        # -------------------------------------------------
        # Resize to input resolution
        # -------------------------------------------------

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0,
        ]

        # -------------------------------------------------
        # Normalize to [0, 1]
        # -------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        range_value = (
            cam_max - cam_min
        )

        if range_value.item() > 1e-8:

            cam = (
                cam - cam_min
            ) / range_value

        else:

            cam = torch.zeros_like(
                cam
            )

        return (
            cam.detach(),
            logits.detach(),
        )

    # =====================================================
    # CLEANUP
    # =====================================================

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

            self.forward_handle = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.remove_hooks()