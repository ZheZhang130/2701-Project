"""Integration test of app rules with a controlled external ML response."""
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ['APP_DATA_DIR'] = tempfile.mkdtemp(prefix='safedrive-test-')
from fastapi.testclient import TestClient
import main


class FakeML:
    label = 1
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def post(self, url, **kwargs):
        assert url.endswith('/v1/models/resnet18/predict')
        assert kwargs['content'] == b'demo-image'
        probabilities = [0.0] * 10
        probabilities[self.label] = 1.0
        import httpx
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'predicted_class': self.label, 'probabilities': probabilities,
            'model_id': 'resnet18', 'model_version': 'test', 'frame_score': float(self.label != 0)})


class AppTest(unittest.TestCase):
    def test_end_to_end_tagging(self):
        client = TestClient(main.app)
        driver = client.post('/api/drivers', json={'name': 'Demo Driver'}).json()
        other = client.post('/api/drivers', json={'name': 'Other Driver'}).json()
        path = f"/api/drivers/{driver['id']}"
        def frame(label):
            FakeML.label = label
            response = client.post(path+'/frames', content=b'demo-image', headers={'Content-Type':'image/jpeg'})
            self.assertEqual(response.status_code, 200, response.text)
            return response.json()
        with patch.object(main.httpx, 'AsyncClient', FakeML):
            for _ in range(4): self.assertFalse(frame(1)['driver']['tagged'])
            self.assertFalse(frame(0)['driver']['tagged'])
            fifth = frame(2)
            self.assertTrue(fifth['driver']['tagged'])
            frame(3)
            dashboard = client.get('/api/dashboard').json()
            self.assertEqual(len(dashboard['notifications']), 1)
            self.assertFalse(client.get('/api/drivers/'+other['id']).json()['tagged'])
            self.assertEqual(client.get(fifth['frame']['image_url']).content, b'demo-image')
            self.assertEqual(len(client.get(path).json()['frames']), 7)
            client.post('/api/notifications/read')
            self.assertTrue(client.get('/api/dashboard').json()['notifications'][0]['read'])
            client.post(path+'/clear')
            self.assertFalse(frame(1)['driver']['tagged'])
            client.put('/api/rule', json={'window':3,'threshold':2})
            self.assertFalse(frame(1)['driver']['tagged'])
            self.assertTrue(frame(1)['driver']['tagged'])
            self.assertEqual(len(client.get('/api/dashboard').json()['notifications']), 2)
            client.post(path+'/clear')
            for label in [1,0,0,0,1]:
                self.assertFalse(frame(label)['driver']['tagged'])
        self.assertEqual(client.put('/api/rule', json={'window':2,'threshold':3}).status_code, 422)
        self.assertEqual(client.post(path+'/frames',content=b'',headers={'Content-Type':'text/plain'}).status_code,422)
        self.assertTrue(main.STATE_FILE.exists())

class HeartbeatTest(unittest.TestCase):
    def test_dashboard_connection_refreshes_launcher_lease(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            heartbeat = Path(directory) / 'heartbeat'
            with patch.dict(os.environ, {'DEMO_HEARTBEAT_FILE': str(heartbeat)}):
                with TestClient(main.app).websocket_connect('/api/demo/heartbeat') as socket:
                    self.assertEqual(socket.receive_text(), 'connected')
                    self.assertTrue(heartbeat.exists())
                    previous = heartbeat.stat().st_mtime_ns
                    socket.send_text('refresh')
                    self.assertEqual(socket.receive_text(), 'connected')
                    self.assertGreaterEqual(heartbeat.stat().st_mtime_ns, previous)


if __name__ == '__main__': unittest.main()
