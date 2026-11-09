"""Synthetic-only regression tests. Never reads production datasets."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from fractions import Fraction
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from common import map55,time_columns,values,fixed55,write_table
from convert import convert
from validate import validate
from read_frame import StandardDataset

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def video(path,fps=5,size=(64,48),n=4,offset=0,points=None,bframes=0):
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=fps);s.width,s.height=size;s.pix_fmt='yuv420p';s.codec_context.thread_count=1;s.options={'crf':'18','preset':'fast','bf':str(bframes)}
        for i in range(n):
            image=np.zeros((size[1],size[0],3),np.uint8);image[:,:,i%3]=50+i*20;image[:,i:i+8,:]=180
            f=av.VideoFrame.from_ndarray(image,format='rgb24');f.pts=offset+(points[i] if points else i);f.time_base=Fraction(1,fps)
            for p in s.encode(f):c.mux(p)
        for p in s.encode():c.mux(p)
def pixels(path):
    with av.open(str(path)) as c:return [hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest() for f in c.decode(video=0)]
def basic(episodes,fps=5,fields=None,mode='none'):
    return {'dataset_id':'synthetic','profile_id':'synthetic:joint-position:v2','data_kind':'video_only' if fields is None else 'robot','robot_type':'synthetic7','fps':fps,'media':{'policy':'native'},'action':{'mode':mode},'state_fields':fields or [],'time':{'kind':'unknown'},'source':{'id':'synthetic-only','revision':'fixture'},'frames':{'base':{'kind':'robot_base','origin':'test origin','axes':'test xyz, right handed'}},'episodes':episodes}
def joint(source='q'):
    return {'source':source,'slot':'right_arm_joint','owner':'synthetic7','quantity':'joint_position','unit':'rad','component_names':[f'joint{i}' for i in range(7)],'provenance':'measured'}

class ToolkitTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='unified55-test-');self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def run_config(self,c,name='output'):
        p=self.p/(name+'.json');p.write_text(json.dumps(c));return Path(convert(p,self.p/name)['output'])
    def test_slot_units_quaternion_and_unknown_gripper(self):
        t=pa.table({'q':[[180.]*6,[0.]*6],'pose':[[100.,200.,300.,1.,0.,0.,0.],[100.,200.,300.,-1.,0.,0.,0.]],'g':[1.,0.]})
        f=joint();f.update(source_unit='deg',component_names=[f'joint{i}' for i in range(6)])
        pose={'source':'pose','slot':'right_eef','quantity':'pose','unit':'m','source_unit':'mm','quaternion_order':'wxyz','frame':'base','reference_point':'tool0','provenance':'measured'}
        grip={'source':'g','slot':'right_gripper','quantity':'opening','unit':'1','provenance':'measured','gripper':{'mode':'closed_01'}}
        s,m=map55(t,[f,pose,grip]);np.testing.assert_allclose(s[0,7:13],np.pi,atol=1e-6);self.assertEqual(m[0,13],0)
        np.testing.assert_allclose(s[:,21:24],[[.1,.2,.3]]*2);np.testing.assert_array_equal(s[:,24:28],[[0,0,0,1]]*2);np.testing.assert_array_equal(s[:,29],[0,1])
        grip['gripper']['mode']='unknown';s,m=map55(t,[grip]);self.assertFalse(m.any());self.assertFalse(s.any())
    def test_gripper_physical_width_and_out_of_range(self):
        t=pa.table({'width':[0.,40.,80.,90.]});f={'source':'width','slot':'right_gripper','quantity':'opening','unit':'1','provenance':'measured','gripper':{'mode':'width','closed':0,'open':80,'unit':'mm'}}
        s,m=map55(t,[f]);np.testing.assert_array_equal(s[:,29],[0,.5,1,0]);np.testing.assert_array_equal(m[:,29],[1,1,1,0])
        f['provenance']='assumed';_,m=map55(t,[f]);self.assertFalse(m.any())
    def test_physical_clock_provenance_and_epoch_ns(self):
        t=pa.table({'clock':np.array([1700000000000000000,1700000000040000000,1700000000084000000],np.int64)})
        a,_=time_columns(t,{'kind':'measured','column':'clock','unit':'ns'},3);np.testing.assert_allclose(a['source_dt'],[.04,.044,0],atol=1e-12)
        a,_=time_columns(t,{'kind':'published','column':'clock','unit':'ns'},3);self.assertFalse(a['mask_source_dt'].any());self.assertFalse(a['source_dt'].any())
    def test_native_video_only_roundtrip_and_source_untouched(self):
        v=self.p/'a.mp4';video(v);original=sha(v);c=basic([{'videos':{'main':'a.mp4'}}]);out=self.run_config(c)
        self.assertTrue(validate(out,decode=True,samples=4)['passed']);self.assertEqual(original,sha(v))
        ds=StandardDataset(out);self.assertEqual(len(ds),4);self.assertFalse(ds[0]['action_unified'].any());self.assertFalse(ds[0]['mask_state'].any());self.assertFalse(ds[0]['mask_language']);self.assertEqual(ds[0]['observation.images.main'].shape,(48,64,3));ds.close()
        target=next((out/'videos').rglob('*.mp4'));self.assertEqual(sha(v),sha(target))
    def test_different_units_preserve_different_specs(self):
        for i,(fps,size) in enumerate([(5,(64,48)),(12,(80,64))]):
            p=self.p/f'v{i}.mp4';video(p,fps,size);out=self.run_config(basic([{'videos':{'main':p.name}}],fps),f'out{i}');info=json.loads((out/'meta/info.json').read_text());self.assertEqual(info['fps'],fps);self.assertEqual(info['features']['observation.images.main']['shape'],[size[1],size[0],3])
    def test_mixed_specs_normalize_retains_frames_times_and_calibration(self):
        episodes=[]
        for i,(fps,size) in enumerate([(25,(64,48)),(30,(96,64))]):
            video(self.p/f'v{i}.mp4',fps,size)
            t=pa.table({'q':[[float(j+i)]*7 for j in range(4)],'clock':np.array([1700000000000000000+round(j/fps*1e9) for j in range(4)],np.int64),'K':[[10.,0,20,0,10,20,0,0,1]]*4})
            write_table(self.p/f'd{i}.parquet',t);episodes.append({'data':f'd{i}.parquet','videos':{'main':f'v{i}.mp4'},'fps':fps,'task':f'task {i}'})
        c=basic(episodes,25,[joint()],'next_state');c['time']={'kind':'measured','column':'clock','unit':'ns'}
        with self.assertRaisesRegex(ValueError,'mixed|Mixed'):self.run_config(c,'native_bad')
        c['fps']=30;c['media']={'policy':'normalize','fps':30,'size':[80,64]};c['passthrough_columns']='all';c['calibration_columns']={'main':{'K':'K'}}
        out=self.run_config(c,'normalized');report=validate(out,True,6);self.assertTrue(report['passed'],report)
        ds=StandardDataset(out);self.assertEqual(len(ds),8);self.assertAlmostEqual(ds[0]['source_dt'],.04,places=10);self.assertAlmostEqual(ds[4]['source_dt'],.033333333,places=9);self.assertAlmostEqual(ds[1]['timestamp']-ds[0]['timestamp'],1/30,places=8)
        np.testing.assert_array_equal(ds[0]['action_unified'][7:14],ds[1]['state_unified'][7:14]);self.assertFalse(ds[3]['mask_action'].any());self.assertFalse(ds[7]['mask_action'].any())
        side=json.loads((out/'meta/episodes.jsonl').read_text().splitlines()[0]);H=np.asarray(side['videos']['main']['image_transform']);K=np.asarray(ds[0]['source.K']).reshape(3,3);np.testing.assert_allclose(np.asarray(ds[0]['observation.calibration.main.K']).reshape(3,3),H@K,atol=1e-5);ds.close()
    def test_source_action_is_not_forced_to_next_state(self):
        video(self.p/'v.mp4');write_table(self.p/'d.parquet',pa.table({'q':[[float(i)]*7 for i in range(4)],'dq':[[.2]*7]*4}))
        c=basic([{'data':'d.parquet','videos':{'main':'v.mp4'},'task':'move'}],5,[joint()],'source');a=joint('dq');a['quantity']='joint_delta';c['action'].update(fields=[a],terminal='keep')
        out=self.run_config(c);ds=StandardDataset(out);np.testing.assert_allclose(ds[0]['action_unified'][7:14],.2);self.assertTrue(ds[3]['mask_action'].any());ds.close()
    def test_vfr_long_offset_and_subset(self):
        video(self.p/'v.mp4',10,points=[0,2,5,9],offset=20000)
        c=basic([{'videos':{'main':{'path':'v.mp4','fps':10}}}],10);out=self.run_config(c);ds=StandardDataset(out,episode_indices=[0]);self.assertGreater(ds[0]['observation.images.main.pts'],1999);self.assertAlmostEqual(ds[3]['timestamp'],.9,places=6);self.assertFalse(ds[0]['mask_source_dt']);ds.close()
    def test_bframe_packet_merge_exact_pixels_and_boundary(self):
        episodes=[];expected=[]
        for i in range(2):
            p=self.p/f'b{i}.mp4';video(p,5,n=6,bframes=2);expected+=pixels(p);episodes.append({'videos':{'main':p.name},'task':str(i)})
        out=self.run_config(basic(episodes));files=list((out/'videos').rglob('*.mp4'));actual=[]
        for f in sorted(files):actual+=pixels(f)
        self.assertEqual(expected,actual);self.assertEqual(len(files),1);ds=StandardDataset(out);self.assertEqual(ds[5]['task'],'0');self.assertEqual(ds[6]['task'],'1');ds.close()
    def test_existing_lerobot_manifest_reuses_shared_video(self):
        from from_lerobot import adapt
        for i in range(2):video(self.p/f'v{i}.mp4')
        source=self.run_config(basic([{'videos':{'main':f'v{i}.mp4'},'task':str(i)} for i in range(2)]),'original')
        profile=basic([],fields=None);p=self.p/'profile.json';p.write_text(json.dumps(profile));manifest=self.p/'adapted.json';adapt(source,p,manifest)
        out=Path(convert(manifest,self.p/'adapted')['output']);self.assertTrue(validate(out,True,8)['passed']);ds=StandardDataset(out);self.assertEqual(ds[3]['task'],'0');self.assertEqual(ds[4]['task'],'1');ds.close()
    def test_data_only_and_no_cross_episode_physical_dt(self):
        write_table(self.p/'d.parquet',pa.table({'q':[[1.]*7,[2.]*7],'ts':[10.,10.2]}))
        c=basic([{'data':'d.parquet','videos':{},'length':2}],5,[joint()],'next_state');c['time']={'kind':'measured','column':'ts','unit':'s'}
        out=self.run_config(c);ds=StandardDataset(out);self.assertEqual(len(ds),2);self.assertAlmostEqual(ds[0]['source_dt'],.2);self.assertFalse(ds[1]['mask_source_dt']);ds.close()
    def test_known_rigid_transform_and_wrong_matrix_rejection(self):
        from pose import transform_absolute_pose
        T=np.eye(4);T[0,3]=2;out,valid=transform_absolute_pose([[1,0,0,1,0,0,0]],T,'wxyz');np.testing.assert_allclose(out,[[3,0,0,0,0,0,1]]);self.assertTrue(valid[0]);T[0,0]=2
        with self.assertRaises(ValueError):transform_absolute_pose([[1,0,0,1,0,0,0]],T,'wxyz')
    def test_validator_catches_wrong_terminal_mask(self):
        video(self.p/'v.mp4');write_table(self.p/'d.parquet',pa.table({'q':[[float(i)]*7 for i in range(4)]}))
        out=self.run_config(basic([{'data':'d.parquet','videos':{'main':'v.mp4'}}],5,[joint()],'next_state'));p=next((out/'data').rglob('*.parquet'));t=pq.ParquetFile(p).read();ma=values(t['mask_action']).copy();ma[-1,7]=1;t=t.set_column(t.schema.get_field_index('mask_action'),'mask_action',fixed55(ma));write_table(p,t)
        report=validate(out,samples=0);self.assertFalse(report['passed']);self.assertTrue(any('terminal' in x for x in report['errors']))

    def test_existing_lerobot_v21_merge_and_split_preservation(self):
        from from_lerobot import adapt
        src=self.p/'legacy';(src/'meta').mkdir(parents=True);(src/'data').mkdir()
        info={'codebase_version':'v2.1','fps':5,'total_episodes':2,'splits':{'train':'0:2'},'chunks_size':1000,'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':{'observation.images.main':{'dtype':'video'}}}
        episodes=[]
        for i in range(2):
            vp=src/info['video_path'].format(episode_chunk=0,video_key='observation.images.main',episode_index=i);vp.parent.mkdir(parents=True,exist_ok=True);video(vp)
            write_table(src/f'data/episode_{i}.parquet',pa.table({'episode_index':[i]*4,'frame_index':list(range(4)),'task_index':[0]*4,'q':[[float(j)]*7 for j in range(4)]}));episodes.append({'episode_index':i,'length':4})
        (src/'meta/info.json').write_text(json.dumps(info));(src/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':'move'})+'\n');(src/'meta/episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
        c=basic([],5,[joint()],'next_state');profile=self.p/'profile.json';profile.write_text(json.dumps(c));manifest=self.p/'v21.json';adapt(src,profile,manifest);out=Path(convert(manifest,self.p/'v21')['output'])
        self.assertEqual(len(list((out/'videos').rglob('*.mp4'))),1);self.assertTrue(validate(out,True,8)['passed'])
        info['splits']={'train':'0:1','test':'1:2'};(src/'meta/info.json').write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError,'multiple/unknown splits'):adapt(src,profile,manifest)
        c['split']='test';profile.write_text(json.dumps(c));adapt(src,profile,manifest);m=json.loads(manifest.read_text());self.assertEqual(m['split'],'test');self.assertEqual([e['source_episode_id'] for e in m['episodes']],[1])

    def test_schema_owner_and_sample_qc_guard(self):
        video(self.p/'v.mp4');write_table(self.p/'d.parquet',pa.table({'q':[[float(i)]*7 for i in range(4)],'bad':[False,True,False,False]}));c=basic([{'data':'d.parquet','videos':{'main':'v.mp4'}}],5,[joint()],'next_state')
        c['state_fields'][0]['owner']='different_robot'
        with self.assertRaisesRegex(ValueError,'belong to this unit'):self.run_config(c,'wrong_owner')
        c['state_fields'][0]['owner']='synthetic7';c['invalid_sample_column']='bad';out=self.run_config(c,'qc');ds=StandardDataset(out);self.assertFalse(ds[1]['mask_state'].any());self.assertFalse(ds[1]['mask_action'].any());self.assertFalse(ds[0]['mask_action'].any());ds.close()
        c['media']['policy']='normalize'
        with self.assertRaisesRegex(ValueError,'required property'):self.run_config(c,'missing_normalize_specs')

    def test_validator_rejects_invalid_clock_mask(self):
        video(self.p/'v.mp4');out=self.run_config(basic([{'videos':{'main':'v.mp4'}}]));p=next((out/'data').rglob('*.parquet'));t=pq.ParquetFile(p).read();t=t.set_column(t.schema.get_field_index('mask_source_timestamp'),'mask_source_timestamp',pa.array([1.]*4,pa.float32()));write_table(p,t)
        report=validate(out,samples=0);self.assertFalse(report['passed']);self.assertTrue(any('unknown source clock' in x for x in report['errors']))

if __name__=='__main__':unittest.main(verbosity=2)
