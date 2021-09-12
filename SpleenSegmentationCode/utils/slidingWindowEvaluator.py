from typing import (Any, Callable, Dict, Mapping, Optional, Sequence, Tuple,
                    Union)

import monai
import torch
from ignite.engine.engine import Engine
from ignite.metrics.metric import Metric
from ignite.utils import convert_tensor
from monai.inferers.utils import sliding_window_inference
from monai.utils.enums import BlendMode


def _prepare_batch(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def create_sliding_window_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
    roi_size=Tuple[int, int],
    mode=BlendMode.CONSTANT,
    sw_batch_size=4,
) -> Engine:
    metrics = metrics or {}

    def _inference(engine: Engine,  batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, mode=mode)

        return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
