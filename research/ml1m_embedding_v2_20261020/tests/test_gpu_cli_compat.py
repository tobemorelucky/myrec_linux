"""CPU-only official argument-parser admission; no model."""
import json,sys,unittest,uuid,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream")]
from common import write_new
from llm.arguments import DataTrainingArguments
from transformers import HfArgumentParser
class CliCompatibility(unittest.TestCase):
    def test_json_alias_is_byte_identical_and_accepted(self):
        p=ROOT/"checks"/"cli_alias"/(uuid.uuid4().hex+".json")
        original=b'{"input":"The movie has attributes","target":""}\n'
        write_new(p,original)
        self.assertEqual(hashlib.sha256(p.read_bytes()).digest(),hashlib.sha256(original).digest())
        (a,)=HfArgumentParser(DataTrainingArguments).parse_args_into_dataclasses(
            ["--train_file",str(p),"--max_source_length","1024"])
        self.assertEqual(a.max_source_length,1024)
        self.assertEqual(a.train_file,str(p))
if __name__=="__main__":unittest.main()
