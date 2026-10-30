"""Final delivery integrity tests, independent of the training environment."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
try:
    import final_package as p
except ImportError:
    p=None

class FinalPackageTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(p,'portable final-package verifier is missing')

    def test_matrix_requires_all_five_seeds_and_selection_policies(self):
        self.api()
        names=p.expected_deep()
        self.assertEqual(len(names),195)
        self.assertEqual(len(set(names)),195)
        self.assertIn('final/pm25_tensor_seed101_mse',names)
        self.assertIn('selection/nasdaq_full_seed71_pearson',names)
        with self.assertRaises(ValueError):p.check_matrix(names[:-1])
        with self.assertRaises(ValueError):p.check_matrix(names+[names[0]])

    def test_manifest_rejects_missing_and_modified_retained_files(self):
        self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);file=root/'known.txt';file.write_bytes(b'verified')
            manifest={'known.txt':{'sha256':hashlib.sha256(b'verified').hexdigest(),'bytes':8}}
            p.verify_files(root,manifest)
            file.write_bytes(b'modified')
            with self.assertRaises(ValueError):p.verify_files(root,manifest)
            file.unlink()
            with self.assertRaises(ValueError):p.verify_files(root,manifest)

    def test_manifest_rejects_absolute_paths_and_parent_traversal(self):
        self.api()
        with tempfile.TemporaryDirectory() as temp:
            for path in ['../outside','/outside','C:/outside','a/../../outside','a\\..\\..\\outside']:
                with self.assertRaises(ValueError):p.checked_path(Path(temp),path)

    def test_inventory_cannot_pass_empty_public_subset_or_omit_best_weights(self):
        self.api()
        with self.assertRaises(ValueError):p.validate_inventory_contract({},public=True)
        files={name:{} for name in p.required_public_files()}
        p.validate_inventory_contract(files,public=True)
        local=dict(files)
        for name in p.expected_deep():
            for file in ['best.pt','config.json','result.json','predictions.csv','history.csv','importance.csv',
                         'importance_history.json','original_artifacts.json','original_complete.json']:
                local['output/reproduction/'+name+'/'+file]={}
        for ds in p.DATASETS:
            for kind in p.KINDS:
                base=f'output/reproduction/classic/{ds}_{kind}/'
                for file in ['config.json','result.json','original_artifacts.json','original_complete.json']:
                    local[base+file]={}
                for i in range(5 if kind in ('rf','xgt') else 1):local[base+f'predictions_{i}.csv']={}
                if kind in ('strx','arimax'):local[base+'fit_details.json']={}
        for ds in p.DATASETS:
            local[f'data/processed/{ds}.npz']={};local[f'data/processed/{ds}.json']={}
        p.validate_inventory_contract(local)
        del local['output/reproduction/final/sml_tensor_seed11/best.pt']
        with self.assertRaises(ValueError):p.validate_inventory_contract(local)

    def test_retention_keeps_selected_models_and_scientific_trajectories(self):
        self.api()
        for name in ['best.pt','predictions.csv','history.csv','importance_history.json','temporal.npz']:
            self.assertTrue(p.retain_deep_file(name))
        for name in ['last.pt','status.json','worker.lock','failure.json','checkpoint.tmp']:
            self.assertFalse(p.retain_deep_file(name))

    def test_deep_inverse_scale_restores_float32_then_uses_float64(self):
        self.api()
        import numpy as np
        values=np.array([.9072873,.12345679],dtype=np.float32)
        csv_decimals=[float(str(x)) for x in values]
        expected=values.astype(np.float64)*17.472+11.146
        actual=p.restore_deep_predictions(csv_decimals,17.472,11.146)
        np.testing.assert_array_equal(actual,expected)

    def test_statistical_replay_uses_saved_column_identity_and_train_only_fill(self):
        self.api()
        import numpy as np
        data={}
        for split,index,x,y in [('train',[2,3],[[1.,10.,9.],[3.,30.,8.]],[.1,.2]),
                                ('val',[5],[[100.,1000.,7.]],[.3]),
                                ('test',[6],[[200.,2000.,6.]],[.4])]:
            data[split+'_idx']=np.asarray(index)
            data[split+'_x']=np.asarray(x)[:,None,:]
            data[split+'_y']=np.asarray(y)
        design,target,first,cut=p.frozen_statistical_grid(data,{'target_column':2,'features':['a','b','y']},[1])
        np.testing.assert_array_equal(design[:,0],[10.,30.,20.,1000.,2000.])
        self.assertTrue(np.isnan(target[2]))
        self.assertEqual((first,cut),(2,3))
        with self.assertRaises(ValueError):p.frozen_statistical_grid(data,{'target_column':2,'features':['a','b','y']},[2])

    def test_strict_statistical_certificate_requires_exact_endpoint_bindings(self):
        self.api()
        import copy
        proofs={f'{ds}_{kind}':{'result_sha256':'r','manifest_sha256':'m','parameters_sha256':'p',
                              'data_sha256':'d','initialization':'exact_diffuse'}
                for ds in p.DATASETS for kind in ['strx','arimax']}
        cert={'status':'PASS','training_only':True,'groups':{
            key:{**value,'status':'PASS','score':1e-8,'boundary_violation':0.,'directions':[{'pass':True}]*3}
            for key,value in proofs.items()}}
        p.check_statistical_certificate(cert,proofs)
        bad=copy.deepcopy(cert);bad['groups']['pm25_strx']['parameters_sha256']='other'
        with self.assertRaises(ValueError):p.check_statistical_certificate(bad,proofs)
        bad=copy.deepcopy(cert);bad['groups']['sml_arimax']['score']=.01
        with self.assertRaises(ValueError):p.check_statistical_certificate(bad,proofs)
        bad=copy.deepcopy(cert);del bad['groups']['nasdaq_arimax']
        with self.assertRaises(ValueError):p.check_statistical_certificate(bad,proofs)

    def test_hpo_minimum_ignores_rejected_trials_without_silently_accepting_missing(self):
        self.api()
        trials=[{'run':'sml_full_00','status':'completed','val_rmse_scaled':.3},
                {'run':'sml_full_01','status':'completed','val_rmse_scaled':.2},
                {'run':'sml_full_02','status':'nonfinite_rejected','val_rmse_scaled':None}]
        selected={'run':'sml_full_01','val_rmse_scaled':.2,'candidate_count':3}
        p.check_hpo_group(trials,selected)
        with self.assertRaises(ValueError):p.check_hpo_group(trials[:-1],selected)
        with self.assertRaises(ValueError):p.check_hpo_group(trials,{**selected,'run':'sml_full_00','val_rmse_scaled':.3})

if __name__=='__main__':unittest.main()
