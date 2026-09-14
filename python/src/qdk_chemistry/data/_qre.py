"""QRE application helpers for chemistry circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Callable
from pathlib import Path

from qdk.estimator import LogicalCounts
from qdk.qre import Trace
from qdk.qre.application import QSharpApplication
from qdk.qre.interop import trace_from_entry_expr_cached


class CachedQSharpApplication(QSharpApplication):
    """Q# application that supports disk caching for callable entry points."""

    cache_key: str

    def __init__(
        self,
        entry_expr: str | Callable | LogicalCounts,
        *,
        cache_key: str,
        args: tuple = (),
        cache_dir: Path | None = None,
    ) -> None:
        if cache_dir is None:
            super().__init__(entry_expr, args=args, use_cache=True)
        else:
            super().__init__(entry_expr, args=args, cache_dir=cache_dir, use_cache=True)
        self.cache_key = cache_key

    def get_trace(self, _parameters: None = None) -> Trace:
        """Return the cached trace, computing and storing it when absent."""
        return trace_from_entry_expr_cached(
            self.entry_expr,
            cache_path=self.cache_dir / f"{self.cache_key}.json",
            use_trace_backend=self.use_trace_backend,
            args=self.args,
        )
