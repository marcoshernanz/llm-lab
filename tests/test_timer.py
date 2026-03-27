from unittest.mock import patch

import pytest

from lib.timer import Timer


def test_stop_returns_elapsed_seconds() -> None:
    timer = Timer()

    with patch("lib.timer.perf_counter", side_effect=[10.0, 12.5]):
        timer.start("training")
        elapsed = timer.stop("training")

    assert elapsed == 2.5
    assert timer.elapsed("training") == 2.5


def test_start_raises_if_timer_is_already_running() -> None:
    timer = Timer()

    with patch("lib.timer.perf_counter", return_value=10.0):
        timer.start("training")

    with pytest.raises(ValueError, match=r"Timer 'training' is already running\."):
        timer.start("training")


def test_measure_context_manager_records_elapsed_time() -> None:
    timer = Timer()

    with patch("lib.timer.perf_counter", side_effect=[20.0, 23.0]):
        with timer.measure("total"):
            pass

    assert timer.elapsed("total") == 3.0
