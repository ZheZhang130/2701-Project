import json
import math
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
from dataclasses import asdict, replace
from PIL import Image
from safedrive.monitoring import DriverMonitor, MonitoringConfig


class FakeClassifier:
    def __init__(self):
        self.score = .99
        self.output = None
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        return [self.output if self.output is not None else {
            'probabilities': [1-self.score, self.score] + [0.]*8, 'model_version': 'fake'}]


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (8, 8))
        self.model = FakeClassifier()
        self.config = MonitoringConfig(high_threshold=.9, low_threshold=.3,
            high_duration_ms=200, low_duration_ms=100, smoothing_tau_ms=1,
            max_gap_ms=150, cooldown_ms=200, representative_limit=2)
        self.monitor = DriverMonitor(self.model, self.config)

    def frame(self, t, score=.99, session='a', frame_id=None, monitor=None):
        self.model.score = score
        return (monitor or self.monitor).process_frame(self.image, session_id=session,
            frame_id=frame_id or str(t), timestamp_ms=t)

    def opened(self, session='a'):
        self.frame(0, session=session)
        self.frame(100, session=session)
        return self.frame(200, session=session).review_event

    def test_sustained_boundary_updates_and_recovery(self):
        event = self.opened()
        self.assertEqual((event.status, event.start_timestamp_ms, event.triggered_timestamp_ms), ('opened', 0, 200))
        update = self.frame(300).review_event
        self.assertEqual((update.status, update.event_id), ('updated', event.event_id))
        self.assertEqual(update.representative_frame_ids, ('0', '100'))
        self.assertEqual(self.frame(400, .01).review_event.status, 'updated')
        closed = self.frame(500, .01).review_event
        self.assertEqual((closed.status, closed.end_timestamp_ms, closed.closure_reason), ('closed', 500, 'recovery'))
        self.assertEqual(len(self.monitor.queue.snapshot()), 1)

    def test_brief_spike_does_not_open(self):
        self.frame(0, .01)
        self.frame(100)
        self.frame(200, .01)
        self.assertIsNone(self.frame(300, .01).review_event)
        self.assertEqual(self.monitor.queue.snapshot(), ())

    def test_high_timer_resets(self):
        self.frame(0)
        self.frame(100, .5)
        self.frame(200)
        self.assertIsNone(self.frame(300).review_event)
        self.assertEqual(self.frame(400).review_event.start_timestamp_ms, 200)

    def test_low_timer_resets_in_hysteresis_band(self):
        self.opened()
        self.frame(300, .01)
        self.assertEqual(self.frame(400, .5).review_event.status, 'updated')
        self.frame(500, .01)
        self.assertEqual(self.frame(600, .01).review_event.status, 'closed')

    def test_time_aware_ema_irregular_intervals(self):
        monitor = DriverMonitor(self.model, replace(self.config, smoothing_tau_ms=100))
        self.frame(0, .1, monitor=monitor)
        a = self.frame(25, .9, monitor=monitor)
        self.assertAlmostEqual(a.smoothed_score, .9 - .8*math.exp(-.25))
        b = self.frame(125, .9, monitor=monitor)
        self.assertAlmostEqual(b.smoothed_score, .9 - .8*math.exp(-1.25))

    def test_gap_closes_at_last_observed_time_resets_smoothing(self):
        event = self.opened()
        result = self.frame(1000, .01)
        self.assertEqual(result.review_event.event_id, event.event_id)
        self.assertEqual((result.review_event.end_timestamp_ms, result.review_event.closure_reason), (200, 'gap'))
        self.assertAlmostEqual(result.smoothed_score, .01)
        self.assertIsNone(self.frame(1100).review_event)

    def test_exact_gap_limit_keeps_continuity(self):
        self.frame(0)
        self.frame(150)
        self.assertEqual(self.frame(300).review_event.status, 'opened')

    def test_gap_does_not_count_as_sustained_evidence(self):
        self.frame(0)
        self.assertIsNone(self.frame(200).review_event)
        self.assertIsNone(self.frame(300).review_event)
        self.assertIsNone(self.frame(400).review_event)
        self.assertIsNone(self.frame(500).review_event)
        self.assertEqual(self.frame(600).review_event.start_timestamp_ms, 400)

    def test_cooldown_does_not_accumulate_high_time(self):
        self.opened()
        self.frame(300, .01)
        self.frame(400, .01)
        self.assertIsNone(self.frame(500).review_event)
        self.assertIsNone(self.frame(600).review_event)
        self.assertIsNone(self.frame(700).review_event)
        self.assertEqual(self.frame(800).review_event.start_timestamp_ms, 600)

    def test_session_isolation(self):
        independent = DriverMonitor(self.model, self.config)
        for t, score in [(0, .1), (100, .99), (200, .99), (300, .99), (400, .01)]:
            a = self.frame(t, score)
            self.frame(t, .01, session='b')
            b = self.frame(t, score, monitor=independent)
            self.assertEqual(a.smoothed_score, b.smoothed_score)
            self.assertEqual(a.review_event.status if a.review_event else None,
                             b.review_event.status if b.review_event else None)
        self.assertTrue(all(x.session_id == 'a' for x in self.monitor.queue.snapshot()))

    def test_close_and_session_id_reuse(self):
        first = self.opened()
        closed = self.monitor.close_session('a')
        self.assertEqual(closed.closure_reason, 'session_closed')
        self.assertIsNone(self.monitor.close_session('a'))
        self.assertNotEqual(self.opened().event_id, first.event_id)

    def test_input_rejection_before_inference_and_without_state_mutation(self):
        self.frame(100)
        calls = self.model.calls
        bad = [dict(timestamp_ms=100), dict(timestamp_ms=-1), dict(timestamp_ms=True),
               dict(timestamp_ms=101.5), dict(timestamp_ms=2**53), dict(frame_id='100'),
               dict(frame_id=' '), dict(session_id=''), dict(frame_id='x'*257),
               dict(image=Image.new('L', (8, 8))), dict(image=None)]
        for patch in bad:
            args = dict(image=self.image, session_id='a', frame_id='next', timestamp_ms=200)
            args.update(patch)
            with self.assertRaises(ValueError):
                self.monitor.process_frame(**args)
        self.assertEqual(calls, self.model.calls)
        self.assertIsNone(self.frame(200, frame_id='next').review_event)

    def test_bad_prediction_is_transactional(self):
        self.frame(0)
        for p in ([float('nan')]*10, [.2]*10, [-.1, 1.1]+[0]*8, [1], None):
            self.model.output = {'probabilities': p, 'model_version': 'fake'}
            with self.assertRaises(ValueError):
                self.frame(100)
        self.model.output = None
        self.frame(100)
        self.assertEqual(self.frame(200).review_event.status, 'opened')

    def test_uncertainty_is_separate_from_alert(self):
        self.model.output = {'probabilities': [.1]*10, 'model_version': 'fake'}
        monitor = DriverMonitor(self.model, MonitoringConfig())
        r = self.frame(0, monitor=monitor)
        self.assertTrue(r.uncertain)
        self.assertIsNone(r.review_event)

    def test_queue_overload_is_transactional_and_retryable(self):
        self.monitor = DriverMonitor(self.model, replace(self.config, queue_capacity=1))
        self.opened()
        self.frame(0, session='b')
        self.frame(100, session='b')
        with self.assertRaises(BufferError):
            self.frame(200, session='b')
        self.monitor.close_session('a')
        event = self.frame(200, session='b').review_event
        self.assertEqual(event.status, 'opened')
        self.assertEqual(event.event_id.rsplit('-', 1)[1], '2')
        self.assertEqual(self.monitor.queue.evicted_closed, 1)
        with self.assertRaises(ValueError):
            self.monitor.queue.acknowledge(event.event_id)
        self.monitor.close_session('b')
        self.assertTrue(self.monitor.queue.acknowledge(event.event_id))
        self.assertFalse(self.monitor.queue.acknowledge(event.event_id))

    def test_capacity_errors_do_not_advance_state(self):
        self.monitor = DriverMonitor(self.model, replace(self.config, max_sessions=1, max_frames_per_session=2))
        self.frame(0)
        with self.assertRaises(BufferError):
            self.frame(0, session='b')
        self.frame(100)
        with self.assertRaises(BufferError):
            self.frame(200)
        self.monitor.close_session('a')
        self.assertIsNone(self.frame(0, session='b').review_event)

    def test_long_stream_has_bounded_refs_and_queue(self):
        self.monitor = DriverMonitor(self.model, replace(self.config, queue_capacity=3))
        for i in range(100):
            self.opened(session=str(i))
            for t in range(300, 10000, 100):
                self.frame(t, session=str(i))
            self.monitor.close_session(str(i))
        self.assertEqual(len(self.monitor._sessions), 0)
        self.assertEqual(len(self.monitor.queue.snapshot()), 3)
        self.assertEqual(self.monitor.queue.evicted_closed, 97)
        self.assertTrue(all(len(e.representative_frame_ids) == 2 for e in self.monitor.queue.snapshot()))
        json.dumps([asdict(e) for e in self.monitor.queue.snapshot()], allow_nan=False)

    def test_bundle_rejects_tampered_files_before_loading_model(self):
        for changed in ('model.pt', 'bundle.json', 'monitoring.json'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                hashes = {}
                for name in ('model.pt', 'bundle.json', 'monitoring.json'):
                    (root / name).write_bytes(b'original')
                    hashes[name] = hashlib.sha256(b'original').hexdigest()
                (root / 'monitoring_bundle.json').write_text(json.dumps({'schema_version': 1, 'sha256': hashes}))
                (root / changed).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    DriverMonitor.load(root, device='cpu')

    def test_inference_failure_does_not_consume_frame(self):
        self.frame(0)
        with patch.object(self.model, 'predict', side_effect=RuntimeError('inference failed')):
            with self.assertRaises(RuntimeError):
                self.frame(100)
        self.frame(100)
        self.assertEqual(self.frame(200).review_event.status, 'opened')

    def test_configuration_validation(self):
        for patch in ({'high_threshold': .1, 'low_threshold': .2}, {'max_gap_ms': 0},
                      {'high_threshold': float('nan')}, {'smoothing_tau_ms': True},
                      {'high_duration_ms': 0}, {'queue_capacity': -1}, {'cooldown_ms': -1}):
            with self.assertRaises(ValueError):
                MonitoringConfig(**patch)


if __name__ == '__main__':
    unittest.main()
