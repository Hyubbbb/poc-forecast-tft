"""horizon-bin 헬퍼 계약 — cold/mid/far 경계 정의.

bin 정의 (plan v0.3.0 R8):
- cold: t+1..t+4 (명절기 이내)
- mid:  t+5..t+12 (~분기)
- far:  t+13..t+52 (장기)
"""

from __future__ import annotations

import pytest

from scripts.forecast_utils import _bin_horizon


class TestBinHorizon:
    @pytest.mark.parametrize("h,expected", [
        (1, "cold"),
        (2, "cold"),
        (4, "cold"),
        (5, "mid"),
        (8, "mid"),
        (12, "mid"),
        (13, "far"),
        (26, "far"),
        (52, "far"),
    ])
    def test_known_boundaries(self, h, expected):
        assert _bin_horizon(h) == expected

    def test_h_zero_or_negative_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            _bin_horizon(0)
        with pytest.raises(ValueError, match="must be >= 1"):
            _bin_horizon(-1)

    def test_h_beyond_52_still_far(self):
        # decoder_len > 52는 사용 안 하지만, 분류 정의는 일관되게 'far'
        assert _bin_horizon(53) == "far"
        assert _bin_horizon(100) == "far"
