import cv2
import numpy as np
import time
import logging
from collections import deque

import multi_event
from multi_event import Camera, denormalize_roi_points

logger = logging.getLogger("CCTV_Aligner")
logger.setLevel(logging.INFO)


# ============================================================
# 1. 튜닝 파라미터
# ============================================================

# 테스트할 때는 1초. 실제 운용에서는 3~10초 추천.
ALIGN_INTERVAL_SEC = 10.0

# ORB 특징점 수. 해상도가 낮거나 장면이 복잡하면 2000~4000 권장.
ORB_FEATURES = 1500

# RANSAC/Homography 성공 조건
MIN_GOOD_MATCHES = 20
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.25
RANSAC_REPROJ_THRESH = 5.0

# tracking reference 업데이트 조건
# 너무 자주 업데이트하면 drift가 커질 수 있으므로 최소 시간 간격을 둔다.
TRACKING_UPDATE_MIN_INTERVAL_SEC = 2.0
TRACKING_UPDATE_MIN_INLIERS = 25
TRACKING_UPDATE_MIN_INLIER_RATIO = 0.35

# anchor 직접 매칭으로 drift를 보정하는 주기
# tracking 누적 drift를 줄이기 위해 가끔 anchor -> current 직접 매칭을 시도한다.
ANCHOR_DIRECT_CHECK_INTERVAL_SEC = 15.0
ANCHOR_DIRECT_MIN_INLIERS = 30
ANCHOR_DIRECT_MIN_INLIER_RATIO = 0.35

# 이상한 homography 방어 조건
MAX_CORNER_SHIFT_RATIO = 0.45      # 프레임 큰 변위 제한
MAX_SCALE_CHANGE = 0.45            # 확대/축소 변화 제한
MAX_PERSPECTIVE_ABS = 0.003        # perspective 성분 과도 방어

# identity 판정 민감도
# 기존 1e-3는 너무 민감해서 ROI가 계속 흔들려 보일 수 있다.
# 단, 실제 적용 여부는 아래 small jitter threshold가 주로 결정한다.
HOMOGRAPHY_IDENTITY_ATOL = 1e-3

# ============================================================
# 작은 jitter 무시 조건
# ============================================================
# 카메라가 실제로 움직이지 않아도 CCTV 압축/노이즈/사람 움직임 때문에
# homography가 매번 아주 조금씩 달라진다.
# 아래 조건보다 작은 변화는 실제 카메라 이동으로 보지 않고 ROI를 고정한다.
MIN_APPLY_TRANSLATION_PX = 5.0     # x/y 이동이 이보다 작으면 무시
MIN_APPLY_ROTATION_DEG = 0.5       # 회전이 이보다 작으면 무시
MIN_APPLY_SCALE_CHANGE = 0.02      # scale 변화가 2%보다 작으면 무시
MIN_APPLY_PERSPECTIVE = 0.0005     # perspective 성분이 아주 작으면 무시

# 실패 시 마지막 정상 ROI를 유지할지 여부
KEEP_LAST_GOOD_ROI_ON_FAILURE = True

# 디버그 출력
DEBUG_ALIGN = True


# ============================================================
# 2. Anchor + Tracking Reference ROI Aligner
# ============================================================

class AnchorTrackingROIAligner:
    """
    anchor reference는 고정하고, tracking reference만 안정적으로 업데이트하는 ROI 정렬기.

    ROI 적용은 항상 anchor 기준 ROI에 H_anchor_to_current를 적용한다.
    tracking reference는 ORB 매칭을 쉽게 만들기 위한 중간 기준일 뿐이다.

    추가 수정:
    - 작은 homography 변화는 jitter로 판단하고 ROI 업데이트를 하지 않는다.
    - 실제 카메라가 크게 움직였을 때만 H_last_good을 갱신한다.
    """

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # anchor reference
        self.anchor_gray = None
        self.anchor_kp = None
        self.anchor_des = None
        self.anchor_shape = None

        # tracking reference
        self.tracking_gray = None
        self.tracking_kp = None
        self.tracking_des = None
        self.tracking_shape = None

        # anchor -> tracking 누적 변환
        self.H_anchor_to_tracking = np.eye(3, dtype=np.float32)

        # 마지막 정상 anchor -> current 변환
        self.H_last_good = np.eye(3, dtype=np.float32)

        self.last_tracking_update_time = 0.0
        self.last_anchor_direct_check_time = 0.0

        self.fail_count = 0
        self.success_count = 0

        self.last_debug = {
            "status": "not_initialized",
            "method": "none",
            "raw_matches": 0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "angle_deg": 0.0,
            "scale": 1.0,
        }

    def _gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _features(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def set_anchor(self, frame):
        """
        현재 화면을 절대 기준 anchor로 등록한다.
        이 프레임은 ROI가 현실 세계 기준으로 정확히 맞아 있는 정상 화면이어야 한다.
        """
        if frame is None:
            self.last_debug["status"] = "set_anchor_failed_no_frame"
            return False

        gray = self._gray(frame)
        kp, des = self._features(gray)

        if des is None or kp is None or len(kp) < MIN_GOOD_MATCHES:
            n = 0 if kp is None else len(kp)
            self.last_debug = {
                "status": f"set_anchor_failed_not_enough_features:{n}",
                "method": "anchor_init",
                "raw_matches": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "angle_deg": 0.0,
                "scale": 1.0,
            }
            print(f"[CCTV_Aligner] anchor 특징점 부족: {n}")
            return False

        self.anchor_gray = gray
        self.anchor_kp = kp
        self.anchor_des = des
        self.anchor_shape = frame.shape[:2]

        # tracking reference도 처음에는 anchor와 동일하게 시작
        self.tracking_gray = gray
        self.tracking_kp = kp
        self.tracking_des = des
        self.tracking_shape = frame.shape[:2]

        self.H_anchor_to_tracking = np.eye(3, dtype=np.float32)
        self.H_last_good = np.eye(3, dtype=np.float32)

        now = time.time()
        self.last_tracking_update_time = now
        self.last_anchor_direct_check_time = now

        self.fail_count = 0
        self.success_count = 0

        self.last_debug = {
            "status": "anchor_set",
            "method": "anchor_init",
            "raw_matches": 0,
            "good_matches": len(kp),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "angle_deg": 0.0,
            "scale": 1.0,
        }

        print(f"[CCTV_Aligner] anchor 기준 프레임 등록 완료: features={len(kp)}")
        logger.info(f"anchor 기준 프레임 등록 완료: features={len(kp)}")
        return True

    def _normalize_H(self, H):
        if H is None:
            return None
        H = H.astype(np.float32)
        if abs(float(H[2, 2])) < 1e-8:
            return None
        return H / H[2, 2]

    def _decompose_homography_rough(self, H):
        """
        Homography에서 대략적인 translation / rotation / scale / perspective 성분을 뽑는다.
        완전한 decomposition은 아니지만 CCTV ROI jitter filtering 용도로 충분하다.
        """
        Hn = self._normalize_H(H)
        if Hn is None:
            return {
                "dx": 0.0,
                "dy": 0.0,
                "angle_deg": 0.0,
                "scale": 1.0,
                "perspective": 0.0,
            }

        dx = float(Hn[0, 2])
        dy = float(Hn[1, 2])

        a = float(Hn[0, 0])
        b = float(Hn[1, 0])
        c = float(Hn[0, 1])
        d = float(Hn[1, 1])

        scale_x = (a * a + b * b) ** 0.5
        scale_y = (c * c + d * d) ** 0.5
        scale = (scale_x + scale_y) / 2.0

        angle_deg = float(np.degrees(np.arctan2(b, a)))
        perspective = max(abs(float(Hn[2, 0])), abs(float(Hn[2, 1])))

        return {
            "dx": dx,
            "dy": dy,
            "angle_deg": angle_deg,
            "scale": scale,
            "perspective": perspective,
        }

    def _is_small_jitter(self, H):
        """
        카메라가 실제로 움직인 게 아니라 feature matching 노이즈 수준이면 True.
        """
        m = self._decompose_homography_rough(H)

        return (
            abs(m["dx"]) < MIN_APPLY_TRANSLATION_PX
            and abs(m["dy"]) < MIN_APPLY_TRANSLATION_PX
            and abs(m["angle_deg"]) < MIN_APPLY_ROTATION_DEG
            and abs(m["scale"] - 1.0) < MIN_APPLY_SCALE_CHANGE
            and abs(m["perspective"]) < MIN_APPLY_PERSPECTIVE
        )

    def _add_motion_debug(self, debug, H):
        m = self._decompose_homography_rough(H)
        debug["dx"] = float(m["dx"])
        debug["dy"] = float(m["dy"])
        debug["angle_deg"] = float(m["angle_deg"])
        debug["scale"] = float(m["scale"])
        debug["perspective"] = float(m["perspective"])
        return debug

    def _match_and_homography(self, src_kp, src_des, dst_kp, dst_des, dst_shape, method_name):
        """
        src reference -> dst current 변환을 계산한다.
        """
        if src_des is None or dst_des is None:
            return None, {
                "status": "descriptor_missing",
                "method": method_name,
                "raw_matches": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "angle_deg": 0.0,
                "scale": 1.0,
            }

        raw = self.matcher.knnMatch(src_des, dst_des, k=2)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            # Lowe ratio test
            if m.distance < 0.75 * n.distance:
                good.append(m)

        debug = {
            "status": "matching",
            "method": method_name,
            "raw_matches": len(raw),
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "angle_deg": 0.0,
            "scale": 1.0,
        }

        if len(good) < MIN_GOOD_MATCHES:
            debug["status"] = f"not_enough_good_matches:{len(good)}"
            return None, debug

        src_pts = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
        if H is None or mask is None:
            debug["status"] = "homography_failed"
            return None, debug

        inliers = int(mask.sum())
        inlier_ratio = inliers / max(1, len(good))

        debug["inliers"] = inliers
        debug["inlier_ratio"] = float(inlier_ratio)
        debug = self._add_motion_debug(debug, H)

        if inliers < MIN_INLIERS:
            debug["status"] = f"not_enough_inliers:{inliers}"
            return None, debug

        if inlier_ratio < MIN_INLIER_RATIO:
            debug["status"] = f"low_inlier_ratio:{inlier_ratio:.2f}"
            return None, debug

        H = H.astype(np.float32)
        ok, reason = self._is_homography_reasonable(H, dst_shape)
        if not ok:
            debug["status"] = reason
            return None, debug

        debug["status"] = "ok"
        return H, debug

    def _is_homography_reasonable(self, H, frame_shape):
        """
        잘못된 매칭으로 ROI가 화면 밖으로 튀는 것을 막기 위한 sanity check.
        """
        if H is None:
            return False, "H_none"

        if not np.isfinite(H).all():
            return False, "H_not_finite"

        h, w = frame_shape[:2]

        corners = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        try:
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        except Exception:
            return False, "corner_transform_failed"

        if not np.isfinite(warped).all():
            return False, "warped_corner_not_finite"

        orig = corners.reshape(-1, 2)
        shift = np.linalg.norm(warped - orig, axis=1)
        mean_shift = float(np.mean(shift))
        max_allowed_shift = max(w, h) * MAX_CORNER_SHIFT_RATIO

        if mean_shift > max_allowed_shift:
            return False, f"rejected_large_shift:{mean_shift:.1f}"

        # scale check: 윗변, 아랫변 길이의 평균으로 대략 검사
        orig_top = np.linalg.norm(orig[1] - orig[0])
        orig_bottom = np.linalg.norm(orig[2] - orig[3])
        warped_top = np.linalg.norm(warped[1] - warped[0])
        warped_bottom = np.linalg.norm(warped[2] - warped[3])

        orig_avg = max(1.0, (orig_top + orig_bottom) / 2.0)
        warped_avg = (warped_top + warped_bottom) / 2.0
        scale = warped_avg / orig_avg

        if scale < (1.0 - MAX_SCALE_CHANGE) or scale > (1.0 + MAX_SCALE_CHANGE):
            return False, f"rejected_scale:{scale:.2f}"

        # perspective 성분이 과하면 잘못된 매칭일 가능성이 큼.
        if abs(float(H[2, 0])) > MAX_PERSPECTIVE_ABS or abs(float(H[2, 1])) > MAX_PERSPECTIVE_ABS:
            return False, f"rejected_perspective:{H[2,0]:.5f},{H[2,1]:.5f}"

        return True, "ok"

    def _should_update_tracking(self, debug, now):
        """
        tracking reference 업데이트 조건.
        성공했다고 매번 업데이트하면 drift가 커질 수 있으므로 조건을 강하게 둔다.
        """
        if debug.get("status") != "ok":
            return False

        if now - self.last_tracking_update_time < TRACKING_UPDATE_MIN_INTERVAL_SEC:
            return False

        if debug.get("inliers", 0) < TRACKING_UPDATE_MIN_INLIERS:
            return False

        if debug.get("inlier_ratio", 0.0) < TRACKING_UPDATE_MIN_INLIER_RATIO:
            return False

        return True

    def _update_tracking_reference(self, frame, kp, des, H_anchor_to_current):
        """
        tracking reference를 현재 프레임으로 업데이트한다.
        단, anchor -> tracking 누적 행렬도 같이 저장하므로 ROI 기준은 anchor에 남아 있다.
        """
        self.tracking_gray = self._gray(frame)
        self.tracking_kp = kp
        self.tracking_des = des
        self.tracking_shape = frame.shape[:2]
        self.H_anchor_to_tracking = H_anchor_to_current.astype(np.float32)
        self.last_tracking_update_time = time.time()

    def estimate_anchor_to_current(self, frame):
        """
        현재 프레임에 대한 H_anchor_to_current를 계산한다.
        성공하면 H를 반환하고, 실패하면 마지막 정상 H 또는 identity를 반환한다.

        수정 포인트:
        - 계산된 H가 작은 jitter면 H_last_good을 갱신하지 않고 identity를 반환한다.
        - 따라서 실제 카메라가 안 움직일 때 ROI가 조금씩 떠다니지 않는다.
        """
        if self.anchor_des is None or self.tracking_des is None:
            self.last_debug["status"] = "not_initialized"
            return np.eye(3, dtype=np.float32), False

        if frame is None:
            self.last_debug["status"] = "no_current_frame"
            return self.H_last_good.copy(), False

        gray = self._gray(frame)
        kp, des = self._features(gray)

        if kp is None or des is None or len(kp) < MIN_GOOD_MATCHES:
            self.fail_count += 1
            self.last_debug = {
                "status": f"current_not_enough_features:{0 if kp is None else len(kp)}",
                "method": "current_features",
                "raw_matches": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "angle_deg": 0.0,
                "scale": 1.0,
            }
            return self.H_last_good.copy() if KEEP_LAST_GOOD_ROI_ON_FAILURE else np.eye(3, dtype=np.float32), False

        now = time.time()

        # ------------------------------------------------------------
        # A. tracking -> current 계산
        # ------------------------------------------------------------
        H_tracking_to_current, dbg_tracking = self._match_and_homography(
            self.tracking_kp,
            self.tracking_des,
            kp,
            des,
            frame.shape[:2],
            method_name="tracking_to_current"
        )

        if H_tracking_to_current is not None:
            H_anchor_to_current = H_tracking_to_current @ self.H_anchor_to_tracking
            H_anchor_to_current = H_anchor_to_current.astype(np.float32)

            ok, reason = self._is_homography_reasonable(H_anchor_to_current, frame.shape[:2])
            dbg_tracking = self._add_motion_debug(dbg_tracking, H_anchor_to_current)

            if ok:
                self.success_count += 1
                self.fail_count = 0
                self.last_debug = dbg_tracking

                # 작은 jitter면 ROI를 업데이트하지 않는다.
                if self._is_small_jitter(H_anchor_to_current):
                    self.last_debug["status"] = "skip_small_jitter_keep_identity"
                    return np.eye(3, dtype=np.float32), True

                self.H_last_good = H_anchor_to_current

                # 조건이 충분히 좋으면 tracking reference 업데이트
                if self._should_update_tracking(dbg_tracking, now):
                    self._update_tracking_reference(frame, kp, des, H_anchor_to_current)
                    self.last_debug["status"] = "ok_tracking_updated"

                # 가끔 anchor 직접 매칭으로 drift 보정 시도
                if now - self.last_anchor_direct_check_time > ANCHOR_DIRECT_CHECK_INTERVAL_SEC:
                    self._try_anchor_direct_correction(frame, kp, des)
                    self.last_anchor_direct_check_time = now

                return self.H_last_good.copy(), True
            else:
                dbg_tracking["status"] = f"anchor_to_current_rejected:{reason}"

        # ------------------------------------------------------------
        # B. tracking 실패 시 anchor -> current 직접 매칭 fallback
        # ------------------------------------------------------------
        H_anchor_direct, dbg_anchor = self._match_and_homography(
            self.anchor_kp,
            self.anchor_des,
            kp,
            des,
            frame.shape[:2],
            method_name="anchor_to_current_fallback"
        )

        if H_anchor_direct is not None:
            self.success_count += 1
            self.fail_count = 0
            self.last_debug = dbg_anchor
            self.last_debug = self._add_motion_debug(self.last_debug, H_anchor_direct)

            # fallback도 작은 jitter면 ROI를 업데이트하지 않는다.
            if self._is_small_jitter(H_anchor_direct):
                self.last_debug["status"] = "skip_small_jitter_anchor_fallback"
                return np.eye(3, dtype=np.float32), True

            self.H_last_good = H_anchor_direct.astype(np.float32)
            self.last_debug["status"] = "ok_anchor_fallback"

            if self._should_update_tracking(dbg_anchor, now):
                self._update_tracking_reference(frame, kp, des, self.H_last_good)
                self.last_debug["status"] = "ok_anchor_fallback_tracking_updated"

            return self.H_last_good.copy(), True

        # ------------------------------------------------------------
        # C. 둘 다 실패
        # ------------------------------------------------------------
        self.fail_count += 1
        self.last_debug = dbg_tracking if dbg_tracking.get("good_matches", 0) >= dbg_anchor.get("good_matches", 0) else dbg_anchor
        self.last_debug["status"] = "failed_keep_last_good:" + str(self.last_debug.get("status"))

        if KEEP_LAST_GOOD_ROI_ON_FAILURE:
            return self.H_last_good.copy(), False
        return np.eye(3, dtype=np.float32), False

    def _try_anchor_direct_correction(self, frame, kp, des):
        """
        tracking 누적 drift를 줄이기 위해 가끔 anchor -> current 직접 매칭을 시도한다.
        직접 매칭이 매우 좋을 때만 H_last_good과 tracking 누적 H를 교정한다.
        """
        H_direct, dbg = self._match_and_homography(
            self.anchor_kp,
            self.anchor_des,
            kp,
            des,
            frame.shape[:2],
            method_name="anchor_direct_drift_check"
        )

        if H_direct is None:
            return False

        if dbg.get("inliers", 0) < ANCHOR_DIRECT_MIN_INLIERS:
            return False

        if dbg.get("inlier_ratio", 0.0) < ANCHOR_DIRECT_MIN_INLIER_RATIO:
            return False

        dbg = self._add_motion_debug(dbg, H_direct)

        # anchor 직접 매칭도 작은 jitter면 drift correction으로 반영하지 않는다.
        if self._is_small_jitter(H_direct):
            self.last_debug = dbg
            self.last_debug["status"] = "anchor_direct_small_jitter_skip"
            return False

        self.H_last_good = H_direct.astype(np.float32)
        self._update_tracking_reference(frame, kp, des, self.H_last_good)
        self.last_debug = dbg
        self.last_debug["status"] = "anchor_direct_corrected_drift"

        print(
            f"[CCTV_Aligner] anchor 직접 매칭으로 drift 교정 | "
            f"inliers={dbg.get('inliers')} ratio={dbg.get('inlier_ratio'):.2f} "
            f"dx={dbg.get('dx', 0.0):.1f} dy={dbg.get('dy', 0.0):.1f} "
            f"angle={dbg.get('angle_deg', 0.0):.2f} scale={dbg.get('scale', 1.0):.3f}"
        )
        return True


# ============================================================
# 3. CameraV2
# ============================================================

class CameraV2(Camera):
    """
    multi_event.Camera를 상속해서 ROI 자동 보정 기능을 추가한다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.aligner = AnchorTrackingROIAligner()
        self.anchor_set = False

        self.base_roi_poly = []
        self.base_roi_lines = []

        self.aligned_roi_poly = []
        self.aligned_roi_lines = []

        self.last_align_time = 0.0
        self.align_status_text = "ALIGN INIT"
        self.align_ok = False
        self.align_shifted = False

        self.status_history = deque(maxlen=10)

    def update_config(self, new_conf):
        """
        설정이 바뀌면 ROI와 anchor 기준도 새로 잡는다.
        """
        super().update_config(new_conf)

        self.aligner = AnchorTrackingROIAligner()
        self.anchor_set = False

        self.base_roi_poly = []
        self.base_roi_lines = []
        self.aligned_roi_poly = []
        self.aligned_roi_lines = []

        self.last_align_time = 0.0
        self.align_status_text = "ALIGN RESET"
        self.align_ok = False
        self.align_shifted = False

        logger.info(f"[CAM:{self.cam_id}] ROI aligner reset after config update")
        print(f"[CCTV_Aligner] CAM {self.cam_id} aligner reset")

    def _update_runtime_roi(self, frame_shape):
        """
        부모 Camera의 ROI 재계산을 막는다.
        ROI는 이 클래스에서 직접 denormalize하고 handler에 주입한다.
        """
        return

    def _initialize_base_roi_if_needed(self, frame):
        """
        cameras.json에 저장된 normalized ROI를 현재 프레임 해상도의 pixel ROI로 변환한다.
        """
        if frame is None:
            return False

        h, w = frame.shape[:2]

        need_init = False
        if self.roi_frame_shape != frame.shape[:2]:
            need_init = True
        if self.roi_poly_norm and not self.base_roi_poly:
            need_init = True
        if self.roi_lines_norm and not self.base_roi_lines:
            need_init = True

        if not need_init:
            return True

        self.base_roi_poly = denormalize_roi_points(self.roi_poly_norm, w, h) if self.roi_poly_norm else []
        self.base_roi_lines = denormalize_roi_points(self.roi_lines_norm, w, h) if self.roi_lines_norm else []

        self.aligned_roi_poly = list(self.base_roi_poly)
        self.aligned_roi_lines = list(self.base_roi_lines)
        self.roi_frame_shape = frame.shape[:2]

        self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)

        print(
            f"[CCTV_Aligner] CAM {self.cam_id} base ROI init | "
            f"poly={len(self.base_roi_poly)} lines={len(self.base_roi_lines)} shape={frame.shape[:2]}"
        )
        logger.info(
            f"[CAM:{self.cam_id}] base ROI init | "
            f"poly={len(self.base_roi_poly)} lines={len(self.base_roi_lines)} shape={frame.shape[:2]}"
        )
        return True

    def _inject_roi_to_handlers(self, roi_poly, roi_lines):
        """
        보정된 ROI를 Camera 자신과 이벤트 핸들러에 주입한다.
        """
        self.roi_poly = roi_poly or []
        self.roi_lines = roi_lines or []

        for ename in self.events:
            if ename not in self.handlers:
                continue

            handler = self.handlers[ename]

            if self.roi_poly and len(self.roi_poly) >= 3:
                handler.roi_poly = np.array(self.roi_poly, dtype=np.int32)
            else:
                handler.roi_poly = np.empty((0, 2), dtype=np.int32)

            if hasattr(handler, "roi_lines"):
                handler.roi_lines = self.roi_lines or []

            # CrossingDetector는 __init__에서 self.lines를 따로 만들어 사용한다.
            # 따라서 roi_lines뿐 아니라 lines도 갱신해야 실제 횡단 판정선이 바뀐다.
            if hasattr(handler, "lines"):
                new_lines = []
                lines = self.roi_lines or []
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        new_lines.append((lines[i], lines[i + 1]))
                handler.lines = new_lines

    def _transform_points(self, pts, H):
        if not pts:
            return []

        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            out = cv2.perspectiveTransform(pts_np, H)
        except Exception as e:
            logger.warning(f"[CAM:{self.cam_id}] ROI transform failed: {e}")
            return pts

        return out.reshape(-1, 2).astype(np.int32).tolist()

    def _update_alignment(self, frame):
        if frame is None:
            return

        self._initialize_base_roi_if_needed(frame)

        if not self.base_roi_poly and not self.base_roi_lines:
            self.align_status_text = "NO ROI"
            self._inject_roi_to_handlers([], [])
            return

        # 최초 정상 프레임을 anchor로 등록
        if not self.anchor_set:
            ok = self.aligner.set_anchor(frame)
            if ok:
                self.anchor_set = True
                self.last_align_time = time.time()
                self.align_status_text = "ANCHOR SET"
                self.align_ok = True
                self.align_shifted = False
            else:
                self.align_status_text = "ANCHOR FAIL"
                self.align_ok = False
            return

        now = time.time()
        if now - self.last_align_time < ALIGN_INTERVAL_SEC:
            return

        H, ok = self.aligner.estimate_anchor_to_current(frame)
        dbg = self.aligner.last_debug

        # 작은 jitter는 estimate_anchor_to_current()에서 identity로 반환된다.
        shifted = not np.allclose(H, np.eye(3, dtype=np.float32), atol=HOMOGRAPHY_IDENTITY_ATOL)

        self.align_ok = ok
        self.align_shifted = shifted

        # 성공했거나, 실패해도 last_good을 유지하는 정책이면 H로 ROI를 계산한다.
        self.aligned_roi_poly = self._transform_points(self.base_roi_poly, H)
        self.aligned_roi_lines = self._transform_points(self.base_roi_lines, H)
        self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)

        status = dbg.get("status", "unknown")
        method = dbg.get("method", "none")
        good = dbg.get("good_matches", 0)
        inliers = dbg.get("inliers", 0)
        ratio = dbg.get("inlier_ratio", 0.0)
        dx = dbg.get("dx", 0.0)
        dy = dbg.get("dy", 0.0)
        angle = dbg.get("angle_deg", 0.0)
        scale = dbg.get("scale", 1.0)

        if ok:
            self.align_status_text = (
                f"ALIGN OK {method} g={good} i={inliers} r={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} a={angle:.2f} s={scale:.3f}"
            )

            if status.startswith("skip_small_jitter") or status == "anchor_direct_small_jitter_skip":
                if DEBUG_ALIGN:
                    print(
                        f"[CCTV_Aligner] CAM {self.cam_id} jitter 무시, ROI 고정 | "
                        f"{method} good={good} inliers={inliers} ratio={ratio:.2f} "
                        f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} scale={scale:.3f} status={status}"
                    )
            elif shifted:
                print(
                    f"[CCTV_Aligner] CAM {self.cam_id} ROI 보정 | "
                    f"{method} good={good} inliers={inliers} ratio={ratio:.2f} "
                    f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} scale={scale:.3f} status={status}"
                )
            elif DEBUG_ALIGN:
                print(
                    f"[CCTV_Aligner] CAM {self.cam_id} no shift | "
                    f"{method} good={good} inliers={inliers} ratio={ratio:.2f} "
                    f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} scale={scale:.3f} status={status}"
                )
        else:
            self.align_status_text = (
                f"ALIGN HOLD {method} g={good} i={inliers} r={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} a={angle:.2f} s={scale:.3f}"
            )
            print(
                f"[CCTV_Aligner] CAM {self.cam_id} 보정 실패, 마지막 정상 ROI 유지 | "
                f"{method} good={good} inliers={inliers} ratio={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} scale={scale:.3f} status={status}"
            )

        self.status_history.append(self.align_status_text)
        self.last_align_time = now

    def run_logic(self, fr, fid, d_main_res, d_helmet_res):
        """
        부모의 AI/event logic 전에 ROI 보정을 먼저 수행한다.
        """
        if fr is None:
            return [], [], {}

        self._update_alignment(fr)
        return super().run_logic(fr, fid, d_main_res, d_helmet_res)

    def draw(self, frame, tracks_main, tracks_helmet, alarms, connected):
        """
        화면에는 기존 multi_event.py의 draw 결과만 표시한다.
        ROI align 상태 텍스트와 추가 디버깅 ROI 선은 카메라 화면에 그리지 않는다.
        보정 로그는 _update_alignment() 내부 print/logger를 통해 터미널에만 출력된다.
        """
        render_frame = super().draw(frame, tracks_main, tracks_helmet, alarms, connected)
        return render_frame


# ============================================================
# 4. Monkey patch 후 기존 main 실행
# ============================================================

if __name__ == "__main__":
    multi_event.Camera = CameraV2
    multi_event.main()
