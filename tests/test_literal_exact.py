"""Literal text and selection regressions; no weights, network or GPU."""
from pathlib import Path
import pytest
from PIL import Image
from waterec_ocr.data_io import scan_pairs
from waterec_ocr.metrics import score_pairs, selection_score, require_exact_selection, EXACT_POLICY, score_price_values

@pytest.mark.parametrize('prediction', ['ab 12', 'AB12', 'AB 12 ', ' AB 12', 'AB-12', 'AB 21'])
def test_every_character_space_case_and_order_matters(prediction):
    assert score_pairs(['AB 12'], [prediction])['correct'] == 0

@pytest.mark.parametrize('gt,pred', [('10.50','10,50'),('10.50','10.5'),('£10.50','10.50'),('€10.50','£10.50'),('é','e\u0301'),('x\n','x')])
def test_no_numeric_or_unicode_or_control_normalization(gt,pred):
    result=score_pairs([gt],[pred]);assert result['correct']==0
    assert set(result)=={'n','correct','exact_accuracy','metric_policy'}

def test_exact_metric_and_legacy_metric_rejected():
    assert score_pairs([' a '],[' a '])['exact_accuracy']==1
    with pytest.raises(ValueError):score_pairs(['a'],['A'],'upper_no_whitespace')
    with pytest.raises(ValueError):score_price_values(['10.0'],['10'])

def test_selection_only_counts_and_earliest_tie():
    first=score_pairs(['A','B'],['A','X']);second=score_pairs(['A','B'],['A','Y'])
    first.update(cer=100,numeric_correct=0);second.update(cer=0,numeric_correct=999)
    assert selection_score(first)==selection_score(second)==(1,0)
    assert not selection_score(second)>selection_score(first)
    assert selection_score(score_pairs(['A','B'],['A','B']))>selection_score(first)
    with pytest.raises(ValueError):require_exact_selection({'best':[1,-.1]})
    assert require_exact_selection({'selection_policy':EXACT_POLICY})

@pytest.mark.parametrize('ending',[b'',b'\n',b'\r\n'])
def test_single_txt_record_framing_preserves_spaces_case(tmp_path,ending):
    Image.new('RGB',(8,8)).save(tmp_path/'a.png')
    (tmp_path/'a.txt').write_bytes(b' aB 10.50 '+ending)
    assert scan_pairs(tmp_path)[0]['label']==' aB 10.50 '

@pytest.mark.parametrize('payload',[b'A\n\n',b'A\r\n\r\n',b'A\nB',b'A\r'])
def test_multiple_or_embedded_txt_newlines_rejected(tmp_path,payload):
    Image.new('RGB',(8,8)).save(tmp_path/'a.png');(tmp_path/'a.txt').write_bytes(payload)
    with pytest.raises(ValueError,match='Single-line TXT'):scan_pairs(tmp_path)


def test_annotator_preserves_literal_label(tmp_path):
    import ast, tempfile, types
    source=Path(__file__).resolve().parents[1]/'annotator/main.py'
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_label');fn.decorator_list=[]
    class Error(Exception):
        def __init__(self,**kwargs):super().__init__(kwargs['detail'])
    env={'Label':object,'image_path':lambda name:tmp_path/name,'tempfile':tempfile,'Path':Path,'IMAGE_DIR':tmp_path,'HTTPException':Error}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
    assert env['save_label']('a.png',types.SimpleNamespace(text=' aB '))['label']==' aB '
    assert (tmp_path/'a.txt').read_text()==' aB '
    with pytest.raises(Error,match='Single-line'):env['save_label']('a.png',types.SimpleNamespace(text='A\nB'))
