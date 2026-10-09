"""Control-flow tests only: fake model/runner/corpus; no recommendation execution."""
import argparse
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main as entry

class ProtocolTests(unittest.TestCase):
    def run_case(self, dev_only, train=1, test_epoch=-1, cache=True):
        calls=[]
        class FakeModel:
            reader="SeqReader"
            def __init__(self,args,corpus): calls.append("model")
            def to(self,device):return self
            def count_variables(self):return 0
            def load_model(self):calls.append("load")
            def actions_after_train(self):calls.append("after")
            class Dataset:
                def __init__(self,model,corpus,phase):
                    if dev_only and phase=="test":raise AssertionError("test Dataset constructed")
                    self.phase=phase;calls.append("dataset:"+phase)
                def prepare(self):calls.append("prepare:"+self.phase)
        class FakeRunner:
            def __init__(self,args):pass
            def train(self,data):
                selfkeys=sorted(data)
                assert selfkeys==(["dev","train"] if dev_only else ["dev","test","train"])
                calls.append("train")
            def print_res(self,data):
                calls.append("eval:"+data.phase);return "fake"
        args=entry.parse_global_args(argparse.ArgumentParser()).parse_args([])
        args.dev_only=dev_only;args.train=train;args.test_epoch=test_epoch
        args.path="synthetic";args.dataset="synthetic";args.regenerate=0;args.gpu=""
        def reader(args):raise AssertionError("Reader executed")
        with patch.object(entry,"args",args,create=True),patch.object(entry,"model_name",FakeModel,create=True), \
             patch.object(entry,"runner_name",FakeRunner,create=True),patch.object(entry,"reader_name",reader,create=True), \
             patch.object(entry.utils,"init_seed"),patch.object(entry.utils,"format_arg_str",return_value=""), \
             patch.object(entry.os.path,"exists",return_value=cache),patch.object(entry.pickle,"load",return_value=object()), \
             patch.object(entry,"open",return_value=io.BytesIO(),create=True):
            entry.main()
        return calls

    def test_dev_only_train(self):
        calls=self.run_case(1)
        self.assertIn("train",calls)
        self.assertNotIn("dataset:test",calls);self.assertNotIn("eval:test",calls)

    def test_legacy_default_preserved(self):
        calls=self.run_case(0)
        self.assertIn("dataset:test",calls);self.assertIn("eval:test",calls)

    def test_dev_only_eval_uses_dev(self):
        calls=self.run_case(1,train=0)
        self.assertIn("eval:dev",calls);self.assertNotIn("train",calls)

    def test_periodic_test_conflict_rejected(self):
        with self.assertRaises(ValueError):self.run_case(1,test_epoch=1)

    def test_cache_missing_refuses_reader_and_writes(self):
        with self.assertRaises(ValueError):self.run_case(1,cache=False)

if __name__=="__main__":unittest.main(verbosity=2)
