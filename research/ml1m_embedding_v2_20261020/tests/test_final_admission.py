"""Final CPU-only guards; no pretrained model, no GPU, no API."""
import json,random,sys,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream"),str(ROOT/"tests")]
from rich_metadata import make_record,audit,title_key,digest_values,read_links
from prepare_prompts import build
from common import cap_text_fields,encode_prompt,require_rental_execution,dropout_feature,prompt_attributes
from test_production import tiny,Token
from training_support import save_adapter_complete,verify_save_restore
class FinalAdmission(unittest.TestCase):
    def sample(self):
        cat={"movie_id":"1","item_id":1,"title":"Toy Story","year":"1995","genres":["Animation","Children's","Comedy"]}
        link={"imdb_id":"tt0114709","tmdb_id":862}
        # Synthetic API structure, not real metadata and never a production input.
        response={"id":862,"imdb_id":"tt0114709","title":"Toy Story","release_date":"1995-01-01",
                  "genres":[{"name":"Animation"}],"overview":"Synthetic test only",
                  "credits":{"crew":[{"name":"Test Director","job":"Director"}],"cast":[]}}
        return cat,link,response
    def test_identity_and_source_digest(self):
        c,l,r=self.sample();rec,reasons=make_record(c,l,r,"TEST")
        self.assertFalse(reasons);self.assertTrue(rec["verified"])
        source=rec["sources"][0]
        self.assertEqual(source["text_sha256"],digest_values({k:rec[k] for k in source["fields"]}))
    def test_ambiguous_movie_rejected(self):
        for key,val in [("id",999),("release_date","2001"),("title","Wrong")]:
            c,l,r=self.sample();r[key]=val
            self.assertIsNone(make_record(c,l,r,"TEST")[0])
    def test_no_metadata_not_ready(self):
        c,_,_=self.sample();records,pending,report=audit({"1":c},{},None)
        self.assertEqual(report["overview_coverage"],0);self.assertFalse(report["ready_for_rich_scft"])
        self.assertEqual(report["uninspected"],1)
    def test_canonical_item_id_not_movie_id(self):
        folder=ROOT/"checks"/"final_cpu"/uuid.uuid4().hex;folder.mkdir(parents=True)
        movies=folder/"movies.dat";movies.write_text("1::Toy Story (1995)::Animation\n2::Other (1996)::Comedy\n")
        rows=build(movies,{"1":2,"2":1})
        self.assertEqual(rows[0]["item"],"2");self.assertEqual(rows[0]["item_id"],1)
    def test_rating_injection_rejected(self):
        folder=ROOT/"checks"/"final_cpu"/uuid.uuid4().hex;folder.mkdir(parents=True)
        movies=folder/"movies.dat";movies.write_text("1::Toy Story (1995)::Animation\n")
        c,l,r=self.sample();rec,_=make_record(c,l,r,"TEST");rec["average_rating"]=5
        with self.assertRaises(ValueError):build(movies,{"1":1},[rec])
    def test_same_caps_encode_and_training(self):
        text="Movie:\ntitle is X; overview is "+"a"*100+";"
        capped=cap_text_fields(text,Token(),{"overview":30})
        one=encode_prompt(text,Token(),64,{"overview":30})[0]
        two=encode_prompt(capped,Token(),64)[0]
        self.assertEqual(one,two)
    def test_adapter_roundtrip(self):
        m=tiny();path=ROOT/"checks"/"final_cpu"/uuid.uuid4().hex
        save_adapter_complete(m,path)
        self.assertTrue(verify_save_restore(m,path)["adapter_round_trip"])
        self.assertLess((path/"adapter_model.bin").stat().st_size,1000000)
    def test_full_model_save_blocked(self):
        from model_patch import LlamaRSEmbV2
        from transformers import LlamaConfig
        base=LlamaRSEmbV2(LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2),pool_type="avg",tau=3.)
        with self.assertRaises(TypeError):save_adapter_complete(base,ROOT/"checks"/"final_cpu"/uuid.uuid4().hex)
    def test_rich_pending_is_blocked_and_no_auto_base(self):
        c=json.loads((ROOT/"configs/scft_rich_pending.json").read_text())
        self.assertIsNone(c["expected_prompt_sha256"])
        self.assertIn("scft_rich",c["output_dir"])
        self.assertIn("superseded",(ROOT/"scripts/route_b.sh").read_text())
    def test_original_prompt_row_order_is_not_item_id(self):
        from common import load_prompts
        path=ROOT/"checks"/"final_cpu"/(uuid.uuid4().hex+".jsonl")
        path.parent.mkdir(parents=True,exist_ok=True)
        rows=[{"input":"B","target":"","item":"20","item_id":2},{"input":"A","target":"","item":"1","item_id":1}]
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        result=load_prompts(path,{"1":1,"20":2})
        self.assertEqual([r["item_id"] for r in result],[1,2])
    def test_no_unapproved_bulk_requests(self):
        from rich_metadata import fetch_samples
        with self.assertRaises(ValueError):
            fetch_samples({str(i):{} for i in range(10)},{},ROOT/"checks"/"never_written",6)
    def test_html_entity_semicolon_is_not_an_attribute(self):
        text="Movie:\ntitle is Et Dieu&#8230;Crea; release year is 1956; genres are Drama;"
        self.assertEqual(len(prompt_attributes(text)),3)
        views=dropout_feature(text,.4,random.Random(42))
        self.assertTrue(all("Et Dieu&#8230;Crea" in v for v in views))
    def test_rental_only_execution_guard(self):
        if str(ROOT).startswith("/home/yyx/"):
            with self.assertRaises(RuntimeError):require_rental_execution()
        else:require_rental_execution()
if __name__=="__main__":unittest.main(verbosity=2)
