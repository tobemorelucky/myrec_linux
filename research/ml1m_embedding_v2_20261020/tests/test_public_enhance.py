"""Public enhancement intake CPU regression; no interactions, APIs or LLM."""
import csv,json,sys,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter")]
from import_public_enhance import checked_rows,convert,valid_plot,validate_published_identity
from common import sha256

class PublicEnhance(unittest.TestCase):
    def fixture(self):
        p=ROOT/"checks/public_enhance_tests"/uuid.uuid4().hex;p.mkdir(parents=True)
        (p/"movies.dat").write_text("1::First (1995)::Comedy\n20::Second (1996)::Drama\n")
        with (p/"text.csv").open("w",newline="") as out:
            w=csv.DictWriter(out,fieldnames=["item","description","genres","tags","reviews"]);w.writeheader()
            for raw,plot,g in [("20","A separate original movie plot about a second family.","Drama"),("1","A valid published movie plot about a first family.","Comedy")]:
                w.writerow({"item":raw,"description":plot,"genres":g,"tags":"must not enter","reviews":"Five stars, shipped fast\nsecond review line"})
        with (p/"links.tsv").open("w",newline="") as out:
            w=csv.DictWriter(out,fieldnames=["movie_id","name","dburl","wiki_url"],delimiter="\t");w.writeheader()
            w.writerow({"movie_id":"1","name":"First (1995)","dburl":"http://dbpedia.org/resource/First","wiki_url":"https://en.wikipedia.org/wiki/First"})
            w.writerow({"movie_id":"20","name":"Second (1996)","dburl":"http://dbpedia.org/resource/Second","wiki_url":"https://en.wikipedia.org/wiki/Second"})
        return p
    def run_convert(self,p):
        return convert(p/"movies.dat",{"1":2,"20":1},p/"text.csv",p/"links.tsv",verify_release=False)
    def test_raw_movie_ids_not_row_positions(self):
        rich,rows,*rest=self.run_convert(self.fixture())
        self.assertEqual([r["movie_id"] for r in rich],["20","1"])
        self.assertEqual([r["item_id"] for r in rows],[1,2])
    def test_review_tags_excluded(self):
        rich,*_=self.run_convert(self.fixture())
        s=json.dumps(rich)
        self.assertNotIn("Five stars",s);self.assertNotIn("must not enter",s)
        self.assertTrue(all("director" not in r and "cast" not in r for r in rich))
    def test_missing_or_invalid_plot_never_filled(self):
        self.assertEqual(valid_plot("")[1],"empty_plot")
        self.assertEqual(valid_plot("No abstract found.")[1],"empty_plot")
        self.assertEqual(valid_plot("Error: request failed")[1],"too_short_plot")
    def test_duplicate_ids_rejected(self):
        p=self.fixture();text=(p/"text.csv").read_text();(p/"text.csv").write_text(text+text.splitlines()[1]+"\n")
        with self.assertRaises(ValueError):checked_rows(p/"text.csv",",","item")
    def test_title_year_conflict_is_pending(self):
        p=self.fixture();(p/"links.tsv").write_text((p/"links.tsv").read_text().replace("First (1995)","First (2001)"))
        rich,rows,missing,empty,pending,genres,report=self.run_convert(p)
        self.assertEqual(len(pending),1);self.assertEqual(len(rich),1)
        self.assertEqual(pending[0]["movie_id"],"1")
    def test_unpinned_file_rejected(self):
        p=self.fixture()
        with self.assertRaises(ValueError):checked_rows(p/"text.csv",",","item","text_ml1m.tsv")
    def test_unknown_publisher_claim_rejected(self):
        rich,*_=self.run_convert(self.fixture())
        self.assertFalse(validate_published_identity(rich[0]))
    def test_real_generated_catalog_coverage(self):
        p=ROOT/"inputs/rich_zenodo_20261020"
        if not (p/"coverage.json").exists():self.skipTest("No real intake on this CPU host")
        report=json.loads((p/"coverage.json").read_text())
        self.assertEqual(report["successful_unique_identity_matches"],3684)
        self.assertEqual(report["verified_overview"],3666)
        self.assertEqual(report["missing_text_records"],21)
        self.assertEqual(report["identity_pending"],1)
        self.assertEqual(sha256(p/"rich_verified.json"),report["rich_source_sha256"])


class TextOnlyTrainingTest(__import__("unittest").TestCase):
    def test_provenance_not_tensorized(self):
        from training_support import scft_text_only
        class Data:
            def __init__(self,columns):self.columns=columns
            def select_columns(self,names):
                return Data({n:self.columns[n] for n in names})
        source={"train":Data({"input":["movie"],"target":[""],
                               "sources":[{"retrieved_at":"2026-10-20"}]})}
        result=scft_text_only(source)
        self.assertEqual(set(result["train"].columns),{"input","target"})
        self.assertEqual(result["train"].columns["input"],["movie"])
        self.assertIn("sources",source["train"].columns)

if __name__=="__main__":unittest.main(verbosity=2)
