"""시연 조건 회귀 검사: 실제 시간 경계, OR, 사람별/방망이별 연속성."""
import unittest
import numpy as np
from event import EventLogic, Hold


FRAME = np.full((800, 900, 3), 80, np.uint8)


def scene(helmet=True, vest=True, bat=False, shift=0):
    p = {"box": [100 + shift, 100, 300 + shift, 600], "score": .95, "class": 2}
    kp = [[0., 0., 0.] for _ in range(17)]
    for index, xy in {5: (160, 220), 6: (240, 220), 7: (260, 300), 8: (140, 300),
                      9: (300, 330), 10: (130, 330), 11: (170, 420), 12: (230, 420)}.items():
        kp[index] = [xy[0] + shift, xy[1], .95]
    pose = dict(p, keypoints=kp)
    return {"person": [p], "pose": [pose],
            "helmet": [{"box": [140 + shift, 110, 230 + shift, 190], "class": 0 if helmet else 1, "score": .95}],
            "vest": [{"box": [140 + shift, 215, 260 + shift, 425], "class": 0, "score": .95}] if vest else [],
            "bat": [{"box": [300 + shift, 270, 320 + shift, 440], "class": 34, "score": .95}] if bat else []}


def run(logic, sample, times):
    events = []
    for t in times:
        events.extend(logic.update(FRAME, sample, float(t))["events"])
    return events


class DemoConditions(unittest.TestCase):
    def test_both_worn_no_alarm(self):
        self.assertEqual(run(EventLogic(), scene(), np.arange(0, 6, .25)), [])

    def test_helmet_only_exact_three_seconds(self):
        logic = EventLogic()
        self.assertEqual(run(logic, scene(helmet=False), np.arange(0, 3, .25)), [])
        events = run(logic, scene(helmet=False), [3])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reasons"], ["no_helmet"])
        self.assertEqual(run(logic, scene(helmet=False), np.arange(3.25, 6, .25)), [])

    def test_vest_only(self):
        events = run(EventLogic(), scene(vest=False), np.arange(0, 3.25, .25))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reasons"], ["no_vest"])

    def test_both_missing_one_safety_event(self):
        events = run(EventLogic(), scene(False, False), np.arange(0, 3.25, .25))
        self.assertEqual(len(events), 1)
        self.assertEqual(set(events[0]["reasons"]), {"no_vest", "no_helmet"})

    def test_alternating_short_violations_do_not_add_up(self):
        logic = EventLogic()
        self.assertEqual(run(logic, scene(False, True), np.arange(0, 2.5, .25)), [])
        self.assertEqual(run(logic, scene(True, False), np.arange(2.5, 5, .25)), [])

    def test_wearing_again_resets_and_rearms(self):
        logic = EventLogic()
        self.assertEqual(len(run(logic, scene(False, True), np.arange(0, 3.25, .25))), 1)
        run(logic, scene(), [3.25])
        self.assertEqual(run(logic, scene(False, True), np.arange(3.5, 6.5, .25)), [])
        self.assertEqual(len(run(logic, scene(False, True), [6.5])), 1)

    def test_held_bat_exact_three_seconds(self):
        logic = EventLogic()
        self.assertEqual(run(logic, scene(bat=True), np.arange(0, 3, .25)), [])
        events = run(logic, scene(bat=True), [3])
        self.assertEqual([e["type"] for e in events], ["weapon"])

    def test_bat_away_from_hands_does_not_trigger(self):
        sample = scene(bat=True)
        sample["bat"][0]["box"] = [360, 630, 400, 760]
        self.assertEqual(run(EventLogic(), sample, np.arange(0, 6, .25)), [])

    def test_missing_wrist_is_not_hand_contact(self):
        sample = scene(bat=True)
        sample["pose"][0]["keypoints"][9][2] = .1
        self.assertEqual(run(EventLogic(), sample, np.arange(0, 6, .25)), [])

    def test_contact_break_restarts_three_seconds(self):
        logic = EventLogic()
        self.assertEqual(run(logic, scene(bat=True), np.arange(0, 2.75, .25)), [])
        run(logic, scene(bat=False), [2.75])
        self.assertEqual(run(logic, scene(bat=True), np.arange(3, 6, .25)), [])
        self.assertEqual(len(run(logic, scene(bat=True), [6])), 1)

    def test_different_bat_does_not_inherit_duration(self):
        logic = EventLogic()
        a, b = scene(bat=True), scene(bat=True)
        a["bat"][0]["box"] = [296, 270, 302, 430]
        b["bat"][0]["box"] = [315, 270, 320, 430]
        self.assertEqual(run(logic, a, np.arange(0, 2.5, .25)), [])
        self.assertEqual(run(logic, b, np.arange(2.5, 5, .25)), [])

    def test_people_have_independent_timers(self):
        logic = EventLogic()
        first, second = scene(vest=False), scene(shift=450)
        both = {k: first[k] + second[k] for k in first}
        events = run(logic, both, np.arange(0, 3.25, .25))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["person_id"], 1)

    def test_pose_and_head_reject_false_person(self):
        sample = scene(vest=False)
        sample["pose"] = sample["helmet"] = []
        self.assertEqual(run(EventLogic(), sample, np.arange(0, 6, .25)), [])

    def test_only_head_visible_is_unknown_not_no_vest(self):
        sample = scene(vest=False)
        for index in (5, 6, 11, 12):
            sample["pose"][0]["keypoints"][index][2] = .05
        logic = EventLogic()
        self.assertEqual(run(logic, sample, np.arange(0, 6, .25)), [])
        result = logic.update(FRAME, sample, 6)
        self.assertEqual(result["people"][0]["vest_status"], "UNKNOWN")

    def test_visible_vest_with_partial_torso_is_positive_evidence(self):
        sample = scene(vest=True)
        for index in (5, 6, 11, 12):
            sample["pose"][0]["keypoints"][index][2] = .05
        result = EventLogic().update(FRAME, sample, 0)
        self.assertEqual(result["people"][0]["vest_status"], "OK")

    def test_person_loss_breaks_streak(self):
        logic = EventLogic()
        run(logic, scene(vest=False), np.arange(0, 2.5, .25))
        run(logic, {k: [] for k in scene()}, [2.5])
        self.assertEqual(run(logic, scene(vest=False), np.arange(2.75, 5.5, .25)), [])

    def test_camera_gap_cannot_complete_timer(self):
        hold = Hold()
        for t in np.arange(0, 2.75, .25):
            self.assertFalse(hold.update(True, t))
        self.assertFalse(hold.update(True, 6))
        self.assertEqual(hold.seconds, 0)

    def test_reconnect_clears_both_event_timers(self):
        logic = EventLogic()
        run(logic, scene(False, False, True), np.arange(0, 2.75, .25))
        logic.reset()
        self.assertEqual(run(logic, scene(False, False, True), np.arange(3, 6, .25)), [])
        self.assertEqual(len(run(logic, scene(False, False, True), [6])), 2)


if __name__ == "__main__":
    unittest.main()
