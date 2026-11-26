import io
import unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from safedrive.api import create_app
from safedrive.monitoring import DriverMonitor, MonitoringConfig


class FakeClassifier:
    metadata = {'model_version': 'fixture-only'}
    def predict(self, images):
        return [dict(probabilities=[.01,.99]+[0.]*8,frame_score=.99,
                     predicted_class=1,model_version='fixture-only') for _ in images]


class ApiTests(unittest.TestCase):
    def setUp(self):
        cfg=MonitoringConfig(high_duration_ms=200,low_duration_ms=100,
            smoothing_tau_ms=1,max_gap_ms=150,max_sessions=1)
        self.app=create_app(api_key='test-key',monitor_factory=lambda:
            {k:DriverMonitor(FakeClassifier(),cfg) for k in ['a','b','c']})
        self.client=TestClient(self.app).__enter__()
        self.client.headers.update({'X-API-Key':'test-key','Content-Type':'image/png'})
        buf=io.BytesIO();Image.new('RGB',(8,8)).save(buf,format='PNG');self.image=buf.getvalue()

    def tearDown(self):self.client.__exit__(None,None,None)

    def frame(self,model='a',time=0,session='trip',frame=None):
        return self.client.post(f'/v1/models/{model}/frames',content=self.image,
            params=dict(session_id=session,frame_id=frame or str(time),timestamp_ms=time))

    def test_auth_and_readiness(self):
        self.assertEqual(self.client.get('/healthz',headers={'X-API-Key':'wrong'}).status_code,200)
        self.assertEqual(self.client.get('/v1/models',headers={'X-API-Key':'wrong'}).status_code,401)
        self.assertEqual(len(self.client.get('/v1/models').json()['models']),3)
        self.assertEqual(self.frame(model='missing').status_code,404)

    def test_state_isolation_and_duplicate_rejection(self):
        for key in ['a','b','c']:self.assertEqual(self.frame(model=key).status_code,200)
        self.assertEqual(self.frame().status_code,409)
        self.assertEqual(self.frame(time=100).status_code,200)
        self.assertEqual(self.frame(time=99).status_code,409)
        self.assertEqual(self.frame(time=200).json()['review_event']['status'],'opened')
        self.assertEqual(self.client.get('/v1/models/b/queue').json()['events'],[])

    def test_queue_lifecycle_ack_and_reuse(self):
        for t in [0,100,200]:result=self.frame(time=t)
        event=result.json()['review_event']['event_id']
        self.assertEqual(self.client.post(f'/v1/models/a/queue/{event}/ack').status_code,409)
        closed=self.client.post('/v1/models/a/sessions/trip/close').json()['review_event']
        self.assertEqual(closed['closure_reason'],'session_closed')
        self.assertTrue(self.client.post(f'/v1/models/a/queue/{event}/ack').json()['removed'])
        self.assertFalse(self.client.post(f'/v1/models/a/queue/{event}/ack').json()['removed'])
        self.assertEqual(self.frame().status_code,200)

    def test_bad_images_and_body_limit_do_not_consume_frame(self):
        path='/v1/models/a/frames?session_id=trip&frame_id=0&timestamp_ms=0'
        self.assertEqual(self.client.post(path,content=b'not an image').status_code,422)
        self.assertEqual(self.client.post(path,content=self.image,headers={'Content-Type':'application/json'}).status_code,415)
        with patch('safedrive.api.MAX_BYTES',16):
            self.assertEqual(self.client.post(path,content=self.image).status_code,413)
        with patch('safedrive.api.MAX_PIXELS',4):
            self.assertEqual(self.client.post(path,content=self.image).status_code,422)
        self.assertEqual(self.frame().status_code,200)

    def test_busy_does_not_consume_frame(self):
        self.app.state.gate.acquire()
        try:self.assertEqual(self.frame().status_code,503)
        finally:self.app.state.gate.release()
        self.assertEqual(self.frame().status_code,200)

    def test_capacity_and_input_limits(self):
        self.assertEqual(self.frame().status_code,200)
        self.assertEqual(self.frame(session='second').status_code,429)
        self.assertEqual(self.frame(time=-1).status_code,422)
        self.assertEqual(self.frame(session=' ').status_code,422)
        self.assertEqual(self.frame(frame='x'*257).status_code,422)

    def test_stateless_prediction_does_not_create_session(self):
        r=self.client.post('/v1/models/a/predict',content=self.image)
        self.assertEqual(r.status_code,200);self.assertEqual(len(r.json()['probabilities']),10)
        self.assertEqual(self.app.state.monitors['a']._sessions,{})

    def test_empty_key_fails_startup(self):
        with self.assertRaises(RuntimeError):
            with TestClient(create_app(api_key='',monitor_factory=lambda:{})):pass


if __name__=='__main__':unittest.main()
