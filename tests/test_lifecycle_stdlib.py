"""Crash/integrity/concurrency contracts, runnable without scientific packages."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from repro import lifecycle as l

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.folder=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def require_api(self):
        self.assertTrue(hasattr(l,'mark_complete'),'durable completion implementation is missing')
    def result(self,status='completed'):
        (self.folder/'result.json').write_text(json.dumps({'status':status}))
        (self.folder/'evidence.txt').write_text('known evidence')
    def test_partial_completion_is_recoverable(self):
        self.require_api(); self.result()
        self.assertFalse(l.is_complete(self.folder))
        l.mark_complete(self.folder)
        self.assertTrue(l.is_complete(self.folder))
    def test_modified_evidence_is_not_silently_reused(self):
        self.require_api(); self.result(); l.mark_complete(self.folder)
        (self.folder/'evidence.txt').write_text('different evidence')
        with self.assertRaises(ValueError): l.is_complete(self.folder)
    def test_nonconverged_fit_does_not_count_as_completed(self):
        self.require_api(); self.result('not_converged'); l.mark_complete(self.folder)
        self.assertFalse(l.is_complete(self.folder,require_converged=True))
    def test_configuration_changes_are_rejected(self):
        self.require_api(); path=self.folder/'config.json'
        l.validate_config(path,{'units':10}); l.validate_config(path,{'units':10})
        with self.assertRaises(ValueError):l.validate_config(path,{'units':20})
    def test_worker_lock_survives_supervisor_independence(self):
        self.require_api(); handle=l.run_lock(self.folder,blocking=False)
        command=[sys.executable,'-c',"from repro.lifecycle import run_lock; run_lock(__import__('pathlib').Path(__import__('sys').argv[1]),blocking=False)",str(self.folder)]
        env={**__import__('os').environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'code')}
        second=subprocess.run(command,env=env,capture_output=True)
        self.assertNotEqual(second.returncode,0)
        handle.close()
        third=subprocess.run(command,env=env,capture_output=True)
        self.assertEqual(third.returncode,0)

if __name__=='__main__': unittest.main()
