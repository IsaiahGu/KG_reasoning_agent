"""No HTTP/GPU: audits observe native failures and preserve native outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hipporag.information_extraction.openie_openai import OpenIE
from supportchain.hipporag_preflight import AuditedOpenIE, PromptLengthError, summarize_audit


class Backend:
    calls=0
    def __init__(self,text='',finish='stop',error=None): self.text,self.finish,self.error=text,finish,error
    def infer(self,*args,**kwargs):
        self.calls+=1
        if self.error: raise self.error
        return self.text,{'finish_reason':self.finish,'prompt_tokens':1,'completion_tokens':2},False


class FakeTokenizer:
    def apply_chat_template(self,*args,**kwargs): return list(range(100))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.path=Path(self.temporary.name)/'audit.jsonl'
    def tearDown(self): self.temporary.cleanup()

    def test_operational_failure_not_empty_success(self):
        backend=Backend(error=TimeoutError('test timeout'))
        openie=AuditedOpenIE(backend,audit_path=self.path)
        with self.assertRaisesRegex(RuntimeError,'NER failed'):
            openie.batch_openie({'doc':{'content':'Alice lives in Paris.'}})
        summary=summarize_audit(self.path)
        self.assertEqual(summary['operational_errors'],1)
        self.assertEqual(summary['empty_outputs'],0)

    def test_native_parse_failure_classified_and_kept(self):
        backend=Backend('this is not JSON')
        native=OpenIE(backend).ner('doc','Alice lives in Paris.')
        observed=AuditedOpenIE(backend,audit_path=self.path).ner('doc','Alice lives in Paris.')
        self.assertEqual(observed.response,native.response)
        self.assertEqual(observed.unique_entities,native.unique_entities)
        summary=summarize_audit(self.path)
        self.assertEqual(summary['parse_errors'],1)
        self.assertEqual(summary['operational_errors'],0)
        self.assertEqual(summary['empty_outputs'],0)

    def test_labelled_entities_keep_their_names(self):
        backend=Backend(json.dumps({'named_entities':[{'text':'Alice','label':'PERSON'},{'text':'Paris','label':'GPE'},{'text':'Alice','label':'PERSON'}]}))
        self.assertEqual(OpenIE(backend).ner('doc','Alice lives in Paris.').unique_entities,[])
        openie=AuditedOpenIE(backend,audit_path=self.path)
        observed=openie.ner('doc','Alice lives in Paris.')
        self.assertEqual(observed.unique_entities,['Alice','Paris'])
        self.assertNotIn('error',observed.metadata)
        summary=summarize_audit(self.path)
        self.assertEqual(summary['ner_passages_recovered_from_labelled_entities'],['doc'])
        self.assertEqual(summary['parse_errors'],0)

    def test_object_without_entity_list_is_kept_empty_up_to_the_limit(self):
        from supportchain.hipporag_preflight import UNPARSED_NER_LIMIT
        backend=Backend(json.dumps({'title':'A film','awards':{'count':7}}))
        openie=AuditedOpenIE(backend,audit_path=self.path)
        for index in range(UNPARSED_NER_LIMIT):
            observed=openie.ner('doc%d'%index,'Alice lives in Paris.')
            self.assertEqual(observed.unique_entities,[])
            self.assertNotIn('error',observed.metadata)
            self.assertTrue(observed.metadata['unparsed_ner_kept_empty'])
        over=openie.ner('one-too-many','Alice lives in Paris.')
        self.assertIn('error',over.metadata)
        summary=summarize_audit(self.path)
        self.assertEqual(len(summary['ner_passages_kept_with_no_entities']),UNPARSED_NER_LIMIT)
        self.assertEqual(summary['parse_errors'],1)

    def test_other_entity_shapes_remain_errors(self):
        for text in ('this is not JSON',json.dumps({'named_entities':[{'name':'Alice'}]}),json.dumps({'named_entities':[['Alice']]})):
            observed=AuditedOpenIE(Backend(text),audit_path=self.path).ner('doc','Alice lives in Paris.')
            self.assertIn('error',observed.metadata,text)
            self.assertEqual(observed.unique_entities,[])

    def test_native_length_repair_not_changed(self):
        backend=Backend('{"named_entities":["Alice"]}',finish='length')
        native=OpenIE(backend).ner('doc','Alice lives in Paris.')
        observed=AuditedOpenIE(backend,audit_path=self.path).ner('doc','Alice lives in Paris.')
        self.assertEqual(observed.unique_entities,native.unique_entities)
        self.assertEqual(observed.response,native.response)
        self.assertEqual(summarize_audit(self.path)['truncated'],1)

    def test_input_overflow_refuses_call_without_truncation(self):
        backend=Backend('{"named_entities":[]}')
        audited=AuditedOpenIE(backend,audit_path=self.path,tokenizer=FakeTokenizer(),context_limit=101,ner_max_tokens=2)
        with self.assertRaises(PromptLengthError): audited.ner('doc','unchanged passage')
        self.assertEqual(backend.calls,0)
        self.assertEqual(summarize_audit(self.path)['operational_errors'],1)

    def test_invalid_triples_are_counted_without_altering_filter(self):
        backend=Backend('{"triples":[["Alice","born in","Paris"],["bad"]]}')
        observed=AuditedOpenIE(backend,audit_path=self.path).triple_extraction('doc','Alice lives in Paris.',['Alice','Paris'])
        self.assertEqual(observed.triples, [['Alice','born in','Paris']])
        self.assertEqual(summarize_audit(self.path)['invalid_triples_dropped'],1)

    def test_failed_truncated_quoted_comma_retains_only_complete_values(self):
        raw='{"triples":[["John Wellborn Root, Jr.","joined","Holabird & Roche"],["John Wellborn Root, Jr.'
        native=OpenIE(Backend(raw,finish='length')).triple_extraction('doc','Original paragraph',['Root'])
        self.assertTrue(native.metadata.get('error'))
        backend=Backend(raw,finish='length')
        result=AuditedOpenIE(backend,audit_path=self.path).triple_extraction('doc','Original paragraph',['Root'])
        self.assertEqual(result.response,raw)
        self.assertEqual(result.triples,[['John Wellborn Root, Jr.','joined','Holabird & Roche']])
        self.assertNotIn('error',result.metadata)
        self.assertEqual(result.metadata['native_parse_error'],native.metadata['error'])
        self.assertEqual(result.metadata['complete_prefix_recovery']['invented_values'],0)
        self.assertEqual(backend.calls,1)
        summary=summarize_audit(self.path)
        self.assertEqual(summary['recovered_truncated_triples'],1)
        self.assertEqual(summary['original_native_parse_failures'],1)
        self.assertEqual(summary['truncated'],1)

    def test_native_success_never_uses_prefix_fallback(self):
        for finish in ('stop','length'):
            raw='{"triples":[["Alice","born in","Paris"]]}'
            with patch('supportchain.hipporag_preflight.recover_triple_prefix',side_effect=AssertionError('must not run')):
                native=OpenIE(Backend(raw,finish=finish)).triple_extraction('doc','text',['Alice'])
                result=AuditedOpenIE(Backend(raw,finish=finish),audit_path=self.path).triple_extraction('doc','text',['Alice'])
            self.assertEqual(result.triples,native.triples)
            self.assertEqual(result.metadata,native.metadata)
            self.assertEqual(result.response,native.response)

    def test_stop_failure_and_malformed_middle_remain_errors(self):
        for raw,finish in (('{"triples":[["A","r","B"],["C', 'stop'),
                           ('{"triples":[["A","r","B"],["C",BAD,"John, Jr.', 'length')):
            native=OpenIE(Backend(raw,finish=finish)).triple_extraction('doc','text',['A'])
            self.assertIn('error',native.metadata)
            result=AuditedOpenIE(Backend(raw,finish=finish),audit_path=self.path).triple_extraction('doc','text',['A'])
            self.assertIn('error',result.metadata)
            self.assertEqual(result.triples,[])
            self.assertNotIn('complete_prefix_recovery',result.metadata)

    def test_fallback_native_filter_rejection_preserves_original_failure(self):
        raw='{"triples":[["A","r","B"],["John, Jr.'
        with patch('supportchain.hipporag_preflight.filter_invalid_triples',side_effect=TypeError('bad complete value')):
            result=AuditedOpenIE(Backend(raw,finish='length'),audit_path=self.path).triple_extraction('doc','text',['A'])
        self.assertIn('error',result.metadata)
        self.assertEqual(result.triples,[])
        self.assertEqual(result.response,raw)
        self.assertIn('prefix_recovery_rejected',result.metadata)


if __name__=='__main__': unittest.main()
