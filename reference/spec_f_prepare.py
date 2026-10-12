"""Prepare Spec F datasets, document splits, exact cloze joins, and token caches.

CPU only. No model training and no chat/instruction formatting.
"""
import csv
from collections import Counter
import json
import os
from pathlib import Path
import random
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from spec_w_common import ROOT,atomic_json,sha,restore_chunk_layout
from spec_w_run import read_jsonl,utc
from spec_w_full import corpus_rows
from audit_wmdp_matches import field_text

OUT=ROOT/'spec_f_outputs'
SRC=ROOT/'spec_w_outputs/recovery_v1'
MODEL=ROOT/'models/deep-ignorance/unfiltered'
STEPS=[0,1,2,4,8,16,32,64,128,256,512,1000]


def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def write_rows(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


class TokenWriter:
    def __init__(self,name):
        self.path=OUT/'cache'/f'{name}.bin';self.f=self.path.open('wb');self.offsets=[0]
    def add(self,ids):
        a=np.asarray(ids,dtype=np.int32);self.f.write(a.tobytes());self.offsets.append(self.offsets[-1]+len(a));return len(self.offsets)-2
    def close(self):
        self.f.flush();os.fsync(self.f.fileno());self.f.close();np.save(self.path.with_suffix('.offsets.npy'),np.asarray(self.offsets,dtype=np.int64));return {'path':str(self.path),'rows':len(self.offsets)-1,'tokens':self.offsets[-1],'bytes':self.path.stat().st_size}


def main():
    if (OUT/'manifest.json').exists():print('Spec F preparation exists; refusing to overwrite frozen data.');return
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    source=Path('/root/.codex/attachments/af02045a-2278-41e8-a64f-ddc8e789a6c4/pasted-text.txt');(OUT/'SPEC_F.md').write_bytes(source.read_bytes())
    assembled={d['doc_id']:d for d in read_jsonl(SRC/'reassembled.jsonl')};docs=[d for d in read_jsonl(SRC/'calibration_inputs.jsonl') if d['doc_id'] in assembled]
    assert len(docs)==27
    counts=Counter(cid for d in docs for cid in d['matched_claim_ids']);order=list(docs);random.Random(0).shuffle(order);val=[]
    for d in order:
        if len(val)==3:break
        if all(counts[c]>1 for c in d['matched_claim_ids']):
            val.append(d['doc_id'])
            for c in d['matched_claim_ids']:counts[c]-=1
    assert len(val)==3
    claims={c['cell']['cell_id']:c['cell'] for d in docs if d['doc_id'] not in val for c in d['matched_canon']}
    qids={q for c in claims.values() for q in json.loads(c['grounding_question_ids'])}
    with (ROOT/'wmdp_rewritten_cas/spec03_outputs/wmdp_questions.csv').open() as h:canonical=list(csv.DictReader(h))
    def key(q,choices,answer):return norm(q),tuple(norm(x) for x in choices),int(answer)
    bykey={}
    for q in canonical:bykey.setdefault(key(q['question'],[q['choice_'+c] for c in 'abcd'],q['correct_index']),[]).append(q)
    evals=[]
    for row in pq.read_table(OUT/'inputs/cloze_compatible.parquet').to_pylist():
        match=bykey.get(key(row['question'],row['choices'],row['answer']),[])
        assert len(match)==1,('Ambiguous/unmapped cloze item',len(match))
        qid=match[0]['question_id'];mapped_claims=[cid for cid,c in claims.items() if qid in json.loads(c['grounding_question_ids'])]
        twins=set()
        for cid in mapped_claims:
            value=norm(claims[cid]['counterfactual_value']).rstrip('.')
            for index,choice in enumerate(row['choices']):
                if norm(choice).rstrip('.')==value:twins.add(index)
        twin=next(iter(twins)) if len(twins)==1 and next(iter(twins))!=row['answer'] else None
        evals.append({'question_id':qid,'question':row['question'],'choices':row['choices'],'gold':int(row['answer']),'matched':qid in qids,'claim_ids':mapped_claims,'twin_choice_index':twin})
    assert len(evals)==1076 and len({e['question_id'] for e in evals})==1076
    write_rows(OUT/'eval_items.jsonl',evals);primary=[e for e in evals if e['matched']];write_rows(OUT/'claim_matched_eval.jsonl',primary)
    eval_info={'full_cloze_items':len(evals),'claim_matched_items':len(primary),'covered_training_claims':len(claims),'grounded_question_ids':sorted(qids),'noncloze_grounded_ids':sorted(qids-{e['question_id'] for e in evals}),'unambiguous_exact_twin_choice_items':sum(e['twin_choice_index'] is not None for e in primary),'join':'NFKC/casefold/whitespace-normalized question AND all ordered choices AND gold; unique match required','small_subset_warning':f'PRIMARY SUBSET ONLY {len(primary)} QUESTIONS; one answer changes accuracy by {100/len(primary):.2f} percentage points. Exploratory only.'}
    atomic_json(OUT/'eval_subset.json',eval_info);print(json.dumps(eval_info,indent=2),flush=True)
    tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'));boundary=tok.token_to_id('<|endoftext|>');assert boundary==0
    originals=TokenWriter('pair_original');twins=TokenWriter('pair_twin');pairs=[];dropped=[];alignment_failures=[]
    task_rows=read_jsonl(SRC/'calibration_tasks.jsonl');saved={(x['doc_id'],x['chunk_index']):x for x in read_jsonl(SRC/'rewrites.jsonl')}
    tasks=[t for t in task_rows if t['doc_id'] in assembled]
    for d in docs:
        ts=sorted([t for t in tasks if t['doc_id']==d['doc_id']],key=lambda t:t['chunk_index'])
        assert ''.join(restore_chunk_layout(t['text'],saved[(t['doc_id'],t['chunk_index'])]['result']['rewrite'])+t['separator'] for t in ts)==assembled[d['doc_id']]['rewrite']
    for t in tasks:
        try:rewritten=restore_chunk_layout(t['text'],saved[(t['doc_id'],t['chunk_index'])]['result']['rewrite'])
        except ValueError as ex:alignment_failures.append({'doc_id':t['doc_id'],'chunk_index':t['chunk_index'],'reason':str(ex)});continue
        a=tok.encode(t['text'],add_special_tokens=False).ids;b=tok.encode(rewritten,add_special_tokens=False).ids
        if max(len(a),len(b))>2046:
            dropped.append({'doc_id':t['doc_id'],'chunk_index':t['chunk_index'],'reason':'over_context_limit_without_truncation','original_tokens':len(a),'twin_tokens':len(b)});continue
        index=originals.add([boundary]+a+[boundary]);assert twins.add([boundary]+b+[boundary])==index
        pairs.append({'pair_index':index,'doc_id':t['doc_id'],'chunk_index':t['chunk_index'],'split':'validation' if t['doc_id'] in val else 'train','original_tokens':len(a)+2,'twin_tokens':len(b)+2,'original_sha256':sha(t['text']),'twin_sha256':sha(rewritten),'unchanged':t['text']==rewritten,'claim_ids':t['claim_ids']})
    caches={'original':originals.close(),'twin':twins.close()};write_rows(OUT/'pairs.jsonl',pairs);write_rows(OUT/'dropped_pairs.jsonl',alignment_failures+dropped)
    split={'train_doc_ids':[d['doc_id'] for d in docs if d['doc_id'] not in val],'validation_doc_ids':val,'seed':0,'rule':'Seeded document shuffle; hold out three documents only when every covered claim remains represented in training. No paper crosses splits.','pair_counts':dict(Counter(p['split'] for p in pairs))};atomic_json(OUT/'document_split.json',split)
    token_writer=TokenWriter('attack');attack_rows=[];heldout_rows={d['row_index'] for d in docs if d['doc_id'] in val};source_manifest=json.loads((SRC/'manifest.json').read_text());attack_docs=0
    # Original text is tokenized once for the identical attack across all arms.
    for index,row in corpus_rows(source_manifest['shards']):
        if index in heldout_rows:continue
        text=field_text(row['abstract'] or '',row['text'] or '','abstract+text');ids=tok.encode(text,add_special_tokens=False).ids
        if not ids:continue
        attack_docs+=1
        for part,start in enumerate(range(0,len(ids),2046)):
            block=ids[start:start+2046]
            i=token_writer.add([boundary]+block+[boundary]);attack_rows.append({'index':i,'row_index':index,'chunk_index':part})
        if index%2000==0:print(f'Pretokenized attack through corpus row {index}',flush=True)
    caches['attack']=token_writer.close();write_rows(OUT/'attack_chunks.jsonl',attack_rows)
    np.save(OUT/'cache/attack_order.npy',np.random.default_rng(0).permutation(len(attack_rows)).astype(np.int64))
    train_pairs=[p['pair_index'] for p in pairs if p['split']=='train'];np.save(OUT/'cache/pair_train_indices.npy',np.asarray(train_pairs,dtype=np.int64))
    manifest={'spec':'F','status':'prepared','created_utc':utc(),'preliminary':True,'spec_sha256':sha(source.read_bytes()),'seed':0,
      'model':{'name':'Deep Ignorance unfiltered','path':str(MODEL),'hf_repo':'EleutherAI/deep-ignorance-unfiltered','revision':'c8df368ff247cb90b62e21e1689260701b3ff25a','config_sha256':sha((MODEL/'config.json').read_bytes()),'tokenizer_sha256':sha((MODEL/'tokenizer.json').read_bytes()),'weight_identity':'Pinned checkpoint revision plus local shard filenames/sizes; no extra 13-GB hash pass for this speed-prioritized pilot','weight_files':{p.name:p.stat().st_size for p in MODEL.glob('model-*.safetensors')}},
      'arms':['baseline','SFT-twin','DPO','grad-diff'],'installation_steps':64,'attack_steps':1000,'logged_attack_steps':STEPS,
      'learning_rate':5e-5,'learning_rate_choice':'Middle of existing Spec R grid, fixed once; no hyperparameter search','optimizer':'AdamW','optimizer_betas':[.9,.999],'optimizer_eps':1e-8,'weight_decay':.01,'schedule':'linear','warmup_ratio':.05,'gradient_clip_norm':1.0,
      'lora_rank':256,'lora_alpha':512,'lora_dropout':0.0,'lora_target_modules':'All transformer linear modules (128), matching Spec R resolution; output embedding excluded as in that grid',
      'parameter_dtype':'float32','gradient_dtype':'float32','forward_autocast':'bfloat16','attention':'sdpa','torch_compile':'skipped: avoid compilation startup for this short exploratory installation',
      'max_sequence_length':2048,'text_format':'plain LM, no chat or instruction wrapper; shared EOT/BOS id 0 before and after each chunk; first-token label masked','tokenizer_config_mismatch_note':'Use tokenizer EOT id 0, not config.json eos_token_id 2, for data padding/boundaries; padding labels are masked.',
      'DPO':{'beta':.1,'chosen':'twin','rejected':'paired original','textual_prompt':'empty','common_prefix':'single BOS/EOT token id 0','reference':'Frozen starting unfiltered checkpoint; log probabilities cached once before installation','logprob_reduction':'sum over completion tokens'},
      'grad_diff':{'lambda':1.0,'loss':'mean token NLL(twin) minus mean token NLL(original)','original_loss_ceiling':20.0,'clamp_policy':'Predeclared hard ceiling at 20 nats/token; any activation is logged explicitly; nonfinite losses park the arm'},
      'attack_initialization':'Merge installed adapter into fp32 base, then attach fresh rank-256 LoRA initialized with seed 0. Baseline gets the same fresh adapter. Reset optimizer/scheduler and use the identical saved real-data order.',
      'batch_policy':'Determine largest fitting power-of-two microbatch up to effective batch 16 using a dry paired backward; shared attack batch across arms. Report actual size and any fallback. No accumulation unless needed.',
      'effective_batch_cap':16,'gpu_schedule':{'visible_devices':[0],'requested_devices':2,'available_devices':1,'deviation':'Only one H100 is visible; use one process on that device and run sequentially. Never share a GPU.','settle_seconds':10,'gate_order':'Baseline evaluation first, then installation+evaluation of all three arms; collect all four pre-attack values before any attack.'},
      'source_data':{'papers':27,'original_30_unreassembled_excluded':3,'chunks_in_excluded_papers':len(task_rows)-len(tasks),'aligned_source_pairs':len(tasks),'alignment_failures_within_27':len(alignment_failures),'over_context_pairs_dropped':len(dropped),'usable_pairs':len(pairs),'unchanged_pairs':sum(p['unchanged'] for p in pairs),'training_pairs':len(train_pairs),'validation_pairs':len(pairs)-len(train_pairs),'source':str(SRC),'known_limitations':'Mixed rewrite configurations and known coverage defects; no claim of validated twin installation.'},
      'split':split,'attack_data':{'source':'cached normal cais/wmdp-bio-forget-corpus','field':'abstract+text','documents':attack_docs,'heldout_documents_excluded':sorted(heldout_rows),'chunks':len(attack_rows),'order':'saved seed-0 permutation, cycled only if exhausted','same_across_arms':True},
      'eval':{**eval_info,'dataset':'EleutherAI/wmdp_bio_cloze','revision':'f7f4d630417c01ae23b9d9845f79e72cf391384e','split':'cloze_compatible','prompt':'Question: {question.strip()}\nAnswer:','continuation':'space plus answer choice text','scorer':'lm_eval 0.4.13 HFLM.loglikelihood; batched, bf16 forward, fp32 softmax','primary_metric':'canonical character-length-normalized cloze accuracy (acc_norm)','pre_attack_belief_shift':'Delta mean normalized-choice probability of the real/gold answer versus baseline; negative means reduced real-answer belief. Also report direct twin-choice probability/margin on uniquely mappable twin options; a real-answer drop alone is not proof of twin installation.','twin_mapping':'Unique exact normalized match of counterfactual value to a non-gold cloze choice; ambiguous/unmatched twins remain unmapped','fp32_tables':'NPZ per question: raw and character-normalized sequence log-likelihood scores, raw and normalized softmax probabilities; stable IDs preserved'},
      'caches':caches,'adapter_retention':'All logged attack steps plus installed adapters; available disk permits retention. No full model copies. Interrupted phases are parked unless a completed phase artifact exists.',
      'source_hashes':{str(p):sha(p.read_bytes()) for p in [SRC/'reassembled.jsonl',SRC/'calibration_tasks.jsonl',OUT/'eval_items.jsonl',OUT/'pairs.jsonl',OUT/'document_split.json']}}
    atomic_json(OUT/'manifest.json',manifest)
    (OUT/'RUN_LOG.md').write_text('# Spec F run log\n\n## SURPRISES\n\n- Only one H100 is visible, so arms run sequentially.\n- Claim-matched cloze evaluation has only 15 questions.\n- The model config EOS ID differs from the tokenizer; tokenizer ID 0 is authoritative for data boundaries.\n- One otherwise aligned pair exceeds the context limit when boundaries are included and is dropped, without truncating either member.\n- Canonical cloze scorer was absent from the environment; pinned lm_eval 0.4.13 is installed in an isolated cache.\n- Learning rate 5e-5 and 64 installation steps were selected once for the preliminary run; no search.\n')
    print(json.dumps({'source_data':manifest['source_data'],'attack_data':manifest['attack_data'],'eval':eval_info},indent=2),flush=True)

if __name__=='__main__':main()
