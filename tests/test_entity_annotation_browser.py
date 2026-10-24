"""Optional real-browser verification over synthetic redacted text only."""
import json,sys,threading,time
import pytest
from pathlib import Path
pytest.importorskip("playwright.sync_api", reason="Optional synthetic browser verification requires Playwright")
pytestmark=pytest.mark.skipif(not Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists(), reason="Local Chrome unavailable")
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from candidate_screener.entity_annotation.ui import create_server
from playwright.sync_api import sync_playwright, expect
import re
class Service:
    def __init__(self): self.saved=[]
    def session(self,a): return {'campaign':{'campaign_id':'synthetic','labels':['KNOWLEDGE','SKILL','TITLE','DEGREE'],'guideline_version':'v1'},'documents':[{'doc_id':'d1'},{'doc_id':'d2'}],'progress':{'reviewed':len(self.saved)}}
    def document(self,d,a): return {'doc_id':d,'text':'😀 Python developer. C++ skills.','revision':0,'spans':[],'status':'untouched','reviewed_all':False,'side':'cv','split':'train'}
    def save(self,p): time.sleep(.3); self.saved.append(p); return {'revision':len(self.saved)}
def test_synthetic_browser_selection_edits_and_save_navigation_guard():
    s=Service();server=create_server(s,port=0);threading.Thread(target=server.serve_forever,daemon=True).start()
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page();page.goto(f'http://127.0.0.1:{server.server_port}');page.locator('#annotator').fill('synthetic');page.locator('#connect').click();expect(page.locator('#text')).to_contain_text('Python')
        def select(needle):
            page.evaluate('''needle=>{const area=document.querySelector('#text'),walker=document.createTreeWalker(area,NodeFilter.SHOW_TEXT);let n;while(n=walker.nextNode()){const i=n.textContent.indexOf(needle);if(i>=0){const r=document.createRange();r.setStart(n,i);r.setEnd(n,i+needle.length);window.getSelection().removeAllRanges();window.getSelection().addRange(r);area.dispatchEvent(new MouseEvent('mouseup'));return;}}throw Error('missing text')}''',needle)
        select('Python');page.keyboard.press('1');assert page.locator('#spans .span').count()==1
        page.locator('#save').click();assert page.locator('#next').is_disabled();page.evaluate("document.querySelector('#next').click()");expect(page.locator('#dirty')).to_have_text('Saved');assert s.saved[-1]['spans']==[{'start':3,'end':9,'label':'KNOWLEDGE'}];assert s.saved[-1]['offset_unit']=='utf16'
        page.locator('#spans select').select_option('SKILL');assert page.locator('#spans select').input_value()=='SKILL';page.locator('#undo').click();assert page.locator('#spans select').input_value()=='KNOWLEDGE'
        page.get_by_role('button',name='Reselect',exact=True).click();select('developer');page.locator('#labels button').nth(2).click();assert 'developer' in page.locator('#spans').text_content()
        page.get_by_role('button',name='Delete',exact=True).click();assert page.locator('#spans .span').count()==0;page.locator('#undo').click();assert page.locator('#spans .span').count()==1
        page.get_by_role('button',name='Delete',exact=True).click();page.locator('#reviewedAll').check();page.locator('#empty').click();expect(page.locator('#dirty')).to_have_text('Saved');assert s.saved[-1]['status']=='reviewed' and s.saved[-1]['spans']==[] and s.saved[-1]['reviewed_all']
        page.locator('#next').click();expect(page.locator('#identity')).to_have_text(re.compile(r"^2/2"));page.locator('#back').click();expect(page.locator('#identity')).to_have_text(re.compile(r"^1/2"))
        browser.close()
    server.shutdown();server.server_close();print('PASS synthetic browser: UTF16 emoji selection, keyboard labels, save, relabel, reselect, delete, undo, reviewed-zero, next/back')
