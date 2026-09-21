"""Replay the production matcher without importing device services or changing settings."""
import json
import argparse
import runpy
import time
from pathlib import Path
import cv2
import numpy as np

D = Path(__file__).resolve().parent
NS = runpy.run_path(str(D.parents[1] / 'tests/test_roi_auto_correct_irfilter.py'), run_name='replay')['NS']


def evaluate(a, b, roi, name, expected=None):
    diagnostics = []
    cv2.setRNGSeed(42)
    started = time.perf_counter()
    H, reason = NS['estimate_alignment_homography'](a, b, [], roi, diagnostics)
    elapsed = time.perf_counter() - started
    samples = NS['roi_validation_samples']([], roi)
    for row in diagnostics:
        src = np.float32(row.get('src', [])).reshape(-1, 2)
        good = np.asarray(row.get('inliers', []), bool)
        accepted = src[good] if len(good) else src[:0]
        row['local_counts'] = [int(np.sum(np.linalg.norm(accepted-p, axis=1) <= 80 * max(a.shape)/640)) for p in samples]
        row['inlier_count'] = int(good.sum())
        if row.get('H') is not None:
            mapped = cv2.perspectiveTransform(samples.reshape(-1, 1, 2), np.float64(row['H'])).reshape(-1, 2)
            row['mapped_samples'] = mapped.tolist()
            canvas = cv2.cvtColor(np.hstack([a, b]), cv2.COLOR_GRAY2BGR)
            for s, t, g in zip(src, row['dst'], good):
                if not g:
                    continue
                cv2.line(canvas, tuple(np.int32(s)), tuple(np.int32(np.array(t)+[a.shape[1], 0])), (0, 210, 0), 1)
                cv2.circle(canvas, tuple(np.int32(s)), 3, (0, 210, 0), -1)
            cv2.polylines(canvas, [np.int32(roi)], False, (255, 255, 0), 2)
            cv2.polylines(canvas, [np.int32(mapped[:len(roi)]+[a.shape[1], 0])], False, (255, 255, 0), 2)
            cv2.putText(canvas, f'{row["method"]}: {"ACCEPTED" if row["accepted"] else "REJECTED"}',
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 210, 0) if row['accepted'] else (0, 0, 255), 2)
            cv2.imwrite(str(D / f'{name}_{row["method"]}.jpg'), canvas)
    result = dict(name=name, accepted=H is not None, reason=reason, seconds=elapsed, attempts=diagnostics)
    if expected is not None and H is not None:
        mapped = cv2.perspectiveTransform(samples.reshape(-1, 1, 2), H)
        truth = cv2.perspectiveTransform(samples.reshape(-1, 1, 2), expected)
        result['max_roi_error_px'] = float(np.max(np.linalg.norm(mapped-truth, axis=2)))
    return result


def weak_roi_scene():
    rng = np.random.default_rng(19)
    a = np.full((480, 640), 128, np.uint8)
    a[:160] = rng.integers(0, 256, (160, 640), dtype=np.uint8)
    # Weak but real, spatially distinct texture competes with a highly textured roof.
    weak = rng.integers(112, 145, (320, 640), dtype=np.uint8)
    a[160:] = cv2.GaussianBlur(weak, (3, 3), 0.6)
    H = np.float64([[0.99, -0.02, 180], [0.015, 1.01, -8], [0.00004, -0.00003, 1]])
    b = cv2.warpPerspective(a, H, (640, 480))
    return a, b, [[100, 270], [100, 420]], H


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, help='Native resolution BASE image; no resizing is performed')
    parser.add_argument('--current', type=Path)
    parser.add_argument('--lines', help='JSON line endpoints in BASE image pixels')
    args = parser.parse_args()
    if args.base or args.current or args.lines:
        if not (args.base and args.current and args.lines):
            parser.error('--base, --current and --lines must be supplied together')
        a, b = [cv2.imread(str(p), 0) for p in (args.base, args.current)]
        if a is None or b is None:
            parser.error('Could not decode input images')
        result = evaluate(a, b, json.loads(args.lines), 'roi_native')
        (D/'roi_native_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(result['reason'])
        raise SystemExit(0)
    a = cv2.imread(str(D/'CAM4_20260907_182506_suspect_anchor_base.jpg'), 0)
    results = []
    for stage in ('182506_suspect', '183507_confirm'):
        b = cv2.imread(str(D/f'CAM4_20260907_{stage}_current.jpg'), 0)
        results.append(evaluate(a, b, [[47, 463], [37, 268]], f'roi_improved_{stage}'))
    a, b, roi, H = weak_roi_scene()
    results.append(evaluate(a, b, roi, 'roi_improved_weak_texture', H))
    # Run the original global extractor independently on the same synthetic scene.
    saved = NS['roi_balanced_features']
    NS['roi_balanced_features'] = lambda *args: ([], None)
    results.append(evaluate(a, b, roi, 'roi_global_weak_texture', H))
    NS['roi_balanced_features'] = saved
    (D/'roi_feature_improvements.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != 'attempts'}))
        for row in result['attempts']:
            print(row['method'], row['matches'], row['inlier_count'], row['local_counts'], row['reason'])
