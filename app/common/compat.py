"""
scikit-learn version compatibility shims.

Import this module before loading any joblib artifacts that were trained on
scikit-learn 1.8.x and are now being loaded on 1.9.x.

Background
----------
scikit-learn 1.8.x packaged the gradient-boosting loss functions as a
standalone ``_loss`` C extension that pickle stored by its bare module name
(``_loss``). In 1.9.x the same code lives under ``sklearn._loss``, and there
is no top-level ``_loss`` module, so unpickling artifacts produced by 1.8.x
raises ``ModuleNotFoundError: No module named '_loss'``.

The fix is to register the new location under the old bare name before
joblib tries to import it during unpickling. This does NOT affect runtime
behaviour -- the same C code is used, just imported through the new path.
"""
from __future__ import annotations

import sys

# Map the old bare name '_loss' to sklearn._loss.loss so artifacts pickled
# by sklearn 1.8.x can still be loaded by 1.9.x.
if "_loss" not in sys.modules:
    try:
        import sklearn._loss.loss as _loss_mod  # noqa: F401
        sys.modules["_loss"] = _loss_mod
    except ImportError:
        pass  # If neither exists the downstream joblib.load will fail with a
              # clear error; we don't want to mask that.
