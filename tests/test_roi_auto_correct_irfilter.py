"""동일한 화각 보정 회귀 시나리오를 multi_event_irfilter.py에도 적용한다."""
from pathlib import Path
import runpy


_suite = runpy.run_path(
    str(Path(__file__).with_name("test_roi_auto_correct_danmal.py")),
    init_globals={"SOURCE_FILENAME": "multi_event_irfilter.py"},
)
RoiAutoCorrectTests = _suite["RoiAutoCorrectTests"]
