"""Create a compact delivery package from verified evidence; no invented results."""
import json
import math
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .engine import ROOT,atomic_json,sha,load_data
from .verify import expected_runs
from .audited_repairs import require_strict_complete

OUT=ROOT/'output/reproduction'; DELIVERY=ROOT.parent/'delivery'
CONFIG=json.loads((ROOT/'configs/reproduction.json').read_text())
PAPER={
 ('pm25','full'):(24.47,15.23),('pm25','tensor'):(24.29,14.87),
 ('sml','full'):(.015,.012),('sml','tensor'):(.009,.006),
 ('nasdaq','full'):(.27,.23),('nasdaq','tensor'):(.09,.07)}


def table(frame):
    # Standalone Markdown, no optional tabulate dependency.
    frame=frame.copy()
    def value(v):
        if isinstance(v,float): return f'{v:.6g}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def collect():
    rows=[]
    scales={ds:float(load_data(ROOT/f'data/processed/{ds}.npz')[0]['y_scale']) for ds in CONFIG['datasets']}
    for folder in expected_runs():
        r=json.loads((folder/'result.json').read_text())
        cfg=json.loads((folder/'config.json').read_text())
        rows.append({'dataset':r['dataset'],'model':r['model'],'objective':r['objective'],'selection':r['selection'],
                     'seed':r['seed'],'rmse_raw':r['test_raw']['rmse'],'mae_raw':r['test_raw']['mae'],
                     'rmse_scaled':r['test_scaled']['rmse'],'mae_scaled':r['test_scaled']['mae'],
                     'persistence_rmse_raw':r['persistence_raw']['rmse'],'persistence_rmse_scaled':r['persistence_raw']['rmse']/scales[r['dataset']],'best_epoch':r['best_epoch'],
                     'epochs':r['epochs_completed'],'parameters':r['parameters'],'seconds':r['seconds'],
                     'units':cfg['units'],'lr':cfg['lr'],'dropout':cfg['dropout'],'l2':cfg['l2'],
                     'features':len(r['selected']),'status':r['status'],'run':folder.name})
    for dataset in CONFIG['datasets']:
        for kind in ['rf','xgt','enet','strx','arimax']:
            r=json.loads((require_strict_complete(dataset,kind)/'result.json').read_text())
            for row in r['runs']:
                rows.append({'dataset':dataset,'model':kind,'objective':'classical','selection':'all','seed':row['seed'],
                             'rmse_raw':row['test_raw']['rmse'],'mae_raw':row['test_raw']['mae'],
                             'rmse_scaled':row['test_scaled']['rmse'],'mae_scaled':row['test_scaled']['mae'],
                             'seconds':r['seconds'],'status':r['status'],'run':f'{dataset}_{kind}',
                             'statistical_repair':r.get('repair',False),'initialization':r.get('initialization','original default')})
    return pd.DataFrame(rows)


def summary(frame):
    records=[]
    for keys,group in frame.groupby(['dataset','model','objective','selection'],dropna=False):
        row=dict(zip(['dataset','model','objective','selection'],keys)); row['n_runs']=len(group)
        for metric in ['rmse_raw','mae_raw','rmse_scaled','mae_scaled']:
            vals=group[metric].to_numpy(float)
            row[metric+'_mean']=vals.mean(); row[metric+'_sd']=vals.std(ddof=1) if len(vals)>1 else 0.
            row[metric+'_se']=row[metric+'_sd']/math.sqrt(len(vals))
        row['all_runs_completed']=bool((group.status=='completed').all()); row['statistical_converged']=row['all_runs_completed'] if row['model'] in ('strx','arimax') else None; records.append(row)
    return pd.DataFrame(records)


def uncertainty_and_stability():
    comparisons=[]; stability=[]; rng=np.random.default_rng(20261018)
    for dataset in CONFIG['datasets']:
        for model in ['tensor','full']:
            frames=[pd.read_csv(OUT/'final'/f'{dataset}_{model}_seed{s}'/'predictions.csv') for s in CONFIG['seeds']]
            errors=np.stack([(f.y_pred-f.y_true).to_numpy()**2 for f in frames])
            persistence=(frames[0].persistence-frames[0].y_true).to_numpy()**2
            delta=errors.mean(0)-persistence; n=len(delta); block=20
            boot=[]
            for _ in range(1000):
                starts=rng.integers(0,n,size=math.ceil(n/block))
                ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
                boot.append(delta[ix].mean())
            comparisons.append({'dataset':dataset,'model':model,'delta_mse_vs_persistence':delta.mean(),
                                'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),
                                'block_valid_windows':block,'bootstrap_repeats':1000})
            ranks=[pd.read_csv(OUT/'final'/f'{dataset}_{model}_seed{s}'/'importance.csv') for s in CONFIG['seeds']]
            corrs=[spearmanr(a.importance,b.importance).statistic for i,a in enumerate(ranks) for b in ranks[i+1:]]
            stability.append({'dataset':dataset,'model':model,'mean_pairwise_spearman':float(np.nanmean(corrs)),
                              'top3_seed11':', '.join(ranks[0].sort_values('importance',ascending=False).feature.head(3))})
    return pd.DataFrame(comparisons),pd.DataFrame(stability)



def generalization_diagnostics():
    rows=[]
    for ds in CONFIG['datasets']:
        d,_=load_data(ROOT/f'data/processed/{ds}.npz')
        item={'dataset':ds,'y_min':float(d['y_min']),'y_scale':float(d['y_scale']),
              'test_scaled_min':float(d['test_y'].min()),'test_scaled_max':float(d['test_y'].max()),
              'test_outside_train_fraction':float(np.mean((d['test_y']<0)|(d['test_y']>1))),
              'train_n':len(d['train_y']),'test_n':len(d['test_y'])}
        for model in ['full','tensor']:
            observations=[]
            for seed in CONFIG['seeds']:
                folder=OUT/'final'/f'{ds}_{model}_seed{seed}'
                r=json.loads((folder/'result.json').read_text());frame=pd.read_csv(folder/'predictions.csv');cfg=json.loads((folder/'config.json').read_text())
                observations.append({'seed':seed,'best_epoch':r['best_epoch'],'epochs':r['epochs_completed'],
                    'val_rmse_scaled':r['val_rmse_scaled'],'test_rmse_scaled':r['test_scaled']['rmse'],
                    'test_mean_error_raw':float((frame.y_pred-frame.y_true).mean()),
                    'prediction_scaled_min':float(frame.y_pred_scaled.min()),'prediction_scaled_max':float(frame.y_pred_scaled.max())})
            item[model]={'selected':{k:cfg[k] for k in ['units','lr','dropout','l2']},'runs':observations}
        rows.append(item)
    evidence={'scope':'descriptive post-evaluation diagnostics; no model or tuning changes','source_sha256':sha(Path(__file__)),'rows':rows}
    atomic_json(OUT/'generalization_diagnostics.json',evidence)
    return evidence


def plots(summary_frame,runs):
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,ds in zip(axes,CONFIG['datasets']):
        d=summary_frame[(summary_frame.dataset==ds)&(summary_frame.selection=='all')].copy()
        labels=d.model+' / '+d.objective
        ax.bar(np.arange(len(d)),d.rmse_scaled_mean,yerr=d.rmse_scaled_se,capsize=2)
        ax.set_xticks(np.arange(len(d)),labels,rotation=80,fontsize=7)
        ax.set_yscale('log'); baseline=float(runs[(runs.dataset==ds)&(runs.objective=='em')&(runs.selection=='all')].persistence_rmse_scaled.iloc[0]); ax.axhline(baseline,color='firebrick',linestyle='--',linewidth=1,label='Persistence'); ax.legend(fontsize=7); ax.grid(axis='y',alpha=.2); ax.set_title(ds); ax.set_ylabel('Test RMSE (train min-max scale; log axis)',fontsize=9)
    fig.tight_layout(); fig.savefig(DELIVERY/'figures/forecast_errors.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,13))
    for row,ds in enumerate(CONFIG['datasets']):
        folder=OUT/'final'/f'{ds}_tensor_seed{CONFIG["seeds"][0]}'
        rank=pd.read_csv(folder/'importance.csv').sort_values('importance',ascending=False).head(10)
        axes[row,0].barh(rank.feature[::-1],rank.importance[::-1]); axes[row,0].set_title(ds+' / Tensor: training posterior importance')
        temp=np.load(folder/'temporal.npz')['temporal']
        features=pd.read_csv(folder/'importance.csv').feature.tolist(); ix=[features.index(v) for v in rank.feature]
        im=axes[row,1].imshow(temp[:,ix].T,aspect='auto',cmap='viridis')
        axes[row,1].set_yticks(range(len(ix)),rank.feature,fontsize=7)
        axes[row,1].set_xticks(range(temp.shape[0]),range(temp.shape[0],0,-1))
        axes[row,1].set_xlabel('Lag relative to last observed hidden state (target lag = this + 1)')
        axes[row,1].set_title('Temporal importance / selected seed 11');fig.colorbar(im,ax=axes[row,1])
    fig.tight_layout();fig.savefig(DELIVERY/'figures/importance_temporal.png',dpi=160);plt.close(fig)


def main():
    verified=json.loads((OUT/'verification.json').read_text())
    if verified['status']!='PASS': raise RuntimeError('report requires independent verification')
    DELIVERY.mkdir(parents=True,exist_ok=True);(DELIVERY/'figures').mkdir(exist_ok=True)
    runs=collect(); agg=summary(runs)
    runs.to_csv(DELIVERY/'seed_metrics.csv',index=False);agg.to_csv(DELIVERY/'summary_metrics.csv',index=False)
    selection=agg[agg.selection!='all'];selection.to_csv(DELIVERY/'variable_selection.csv',index=False)
    uncertain,stable=uncertainty_and_stability()
    uncertain.to_csv(DELIVERY/'uncertainty.csv',index=False);stable.to_csv(DELIVERY/'interpretation_stability.csv',index=False)
    data=[]
    for ds in CONFIG['datasets']:
        info=json.loads((ROOT/'data/processed'/f'{ds}.json').read_text())
        data.append({'dataset':ds,'raw_rows':info['raw_rows'],'variables':len(info['features']),
                     'window':info['depth'],'train':info['splits']['train']['n'],'val':info['splits']['val']['n'],
                     'test':info['splits']['test']['n'],'dropped_windows':info['invalid_or_segment_crossing_windows']})
    data=pd.DataFrame(data);data.to_csv(DELIVERY/'data_audit.csv',index=False)
    paper=[]
    for (ds,model),(rmse,mae) in PAPER.items():
        our=agg[(agg.dataset==ds)&(agg.model==model)&(agg.objective=='em')&(agg.selection=='all')].iloc[0]
        paper.append({'dataset':ds,'model':model,'paper_rmse':rmse,'paper_mae':mae,
                      'our_rmse_raw':our.rmse_raw_mean,'our_rmse_scaled':our.rmse_scaled_mean,
                      'directly_comparable':False,'reason':'paper scaling and/or exact preprocessing not fully specified'})
    paper=pd.DataFrame(paper);paper.to_csv(DELIVERY/'paper_comparison.csv',index=False)
    plots(agg,runs)
    main=agg[(agg.selection=='all')&(agg.objective=='em')]
    display=agg[agg.selection=='all'][['dataset','model','objective','n_runs','rmse_raw_mean','rmse_raw_se','mae_raw_mean','rmse_scaled_mean','all_runs_completed']]
    search=[]
    for p in sorted((OUT/'selected').glob('*.json')):
        s=json.loads(p.read_text());search.append({'combination':p.stem,'units/lr/dropout/L2':str(s['params']),
                                                 'best_val_rmse_scaled':s['val_rmse_scaled'],'candidates':s['candidate_count']})
    failures=[]
    for p in sorted((OUT/'hpo').glob('*/invalid_candidate.json')):
        f=json.loads(p.read_text());failures.append({'run':p.parent.name,'error':f['error']})
    findings=[]
    for ds in CONFIG['datasets']:
        rr=runs[(runs.dataset==ds)&(runs.objective=='em')&(runs.selection=='all')]
        persistence=float(rr.persistence_rmse_raw.iloc[0])
        for model in ['tensor','full']:
            v=rr[rr.model==model];mean=float(v.rmse_raw.mean());gain=100*(persistence-mean)/persistence
            word='优于' if gain>0 else '差于或持平'
            findings.append(f'- **{ds} / IMV-{model}**：原单位RMSE均值 {mean:.6g}，persistence为 {persistence:.6g}，相对改善 {gain:.2f}%；本协议下{word}上一观测值基线。5次重复的每一个结果均保留，没有筛去差seed。')
    comparative=[];selection_findings=[];stability_findings=[]
    for ds in CONFIG['datasets']:
        all_rows=agg[(agg.dataset==ds)&(agg.selection=='all')].sort_values('rmse_raw_mean')
        winner=all_rows.iloc[0]
        comparative.append(f"- **{ds} 全模型比较**：本协议均值最低的是 {winner.model}/{winner.objective}，RMSE={winner.rmse_raw_mean:.6g}；这是测试汇总后的描述，未据此回头更改训练或搜索。")
        for model in ['tensor','full']:
            em=all_rows[(all_rows.model==model)&(all_rows.objective=='em')].iloc[0]
            mse=all_rows[(all_rows.model==model)&(all_rows.objective=='mse')].iloc[0]
            gain=100*(mse.rmse_raw_mean-em.rmse_raw_mean)/mse.rmse_raw_mean
            comparative.append(f"  {model}：EM相对相同结构MSE的RMSE改善 {gain:.2f}%（正数为EM更好），EM={em.rmse_raw_mean:.6g}、MSE={mse.rmse_raw_mean:.6g}。")
            for policy in ['importance','pearson']:
                chosen=selection[(selection.dataset==ds)&(selection.model==model)&(selection.selection==policy)].iloc[0]
                delta=100*(chosen.rmse_raw_mean-em.rmse_raw_mean)/em.rmse_raw_mean
                selection_findings.append(f"- {ds}/{model}/{policy}：保留约一半变量后RMSE={chosen.rmse_raw_mean:.6g}，比全变量变化 {delta:+.2f}%（负数为改善）。")
    for row in stable.itertuples():
        word='一致性有限，单seed变量解释应保守' if row.mean_pairwise_spearman<.5 else '有一定跨seed一致性，仍不能解释为因果'
        stability_findings.append(f"- {row.dataset}/{row.model}：平均排名Spearman={row.mean_pairwise_spearman:.4f}；{word}。")
    diagnostic=generalization_diagnostics()
    generalization_rows=[]
    for item in diagnostic['rows']:
        for model in ['full','tensor']:
            observed=item[model]['runs']
            generalization_rows.append({'dataset':item['dataset'],'model':model,'target_outside_train_pct':100*item['test_outside_train_fraction'],
                'mean_val_rmse_scaled':float(np.mean([r['val_rmse_scaled'] for r in observed])),
                'mean_test_rmse_scaled':float(np.mean([r['test_rmse_scaled'] for r in observed])),
                'mean_test_bias_raw':float(np.mean([r['test_mean_error_raw'] for r in observed])),
                'runs_at_epoch_cap':sum(r['epochs']==CONFIG['epochs'] for r in observed)})
    statistical_audit=json.loads((OUT/'strict_final_coordinate_audit.json').read_text())
    if statistical_audit['status']!='PASS' or len(statistical_audit['groups'])!=6:raise RuntimeError('report requires six-group final numerical audit')
    statistical_rows=[]
    for key,row in statistical_audit['groups'].items():
        statistical_rows.append({'group':key,'score_max':row['score'],'boundary_violation':row['boundary_violation'],'initialization':row['initialization'],'status':row['status']})
    nonconverged=agg[agg.model.isin(['strx','arimax']) & ~agg.all_runs_completed]
    publication_path=ROOT.parent/'publication.json'
    publication=json.loads(publication_path.read_text()) if publication_path.exists() else {}
    publication_note=('复现代码及实验快照发布commit：`'+publication['code_commit']+'`。本报告随其后文档提交发布；固定代码commit可用于来源追溯。') if publication.get('code_commit') else '发布commit尚未生成，当前为待发布审阅稿。'
    report=f'''# IMV-LSTM 论文复现实验报告

> 这是基于真实运行与独立checkpoint核验的最终实验报告。对象：Guo, Lin & Antulov-Fantulin, *Exploring Interpretable LSTM Neural Networks over Multi-Variable Data*, ICML 2019。完成范围是三套公开数据上的方法、基线和解释实验；PLANT原始快照未取得，不能称为四数据集逐项数值完全复现。表格内“100%”若出现仅指预登记公开数据计划完成率，绝不表示达到论文全部数值。

## 1. 主要实验发现及其含义

三套公开数据的预登记实验及严格核验均已完成，但**未重现论文的普遍性能优势**。Full仅在PM2.5上较persistence改善约4.53%；Tensor在三套数据上均明显落后。本次结果不能写成“论文所有主要结论均得到验证”。

{chr(10).join(findings)}

{chr(10).join(comparative)}

这些比较检验“变量级hidden + mixture attention能否提高当前公开数据协议下的一步预测”。必须同时看简单persistence、LSTM、DUAL、RETAIN、传统机器学习和不同seed的波动。单个最优seed或单个数据集不足以证明整体优势；预测优势也不意味着注意力给出因果关系。若某模型不优于基线，这是有效的负结果，未通过改测试划分、使用未来数据或挑seed来美化。

## 2. 交付内容与证据边界

- `summary_metrics.csv`：全变量/变量选择各模型的均值、样本SD和SE；`seed_metrics.csv`保留每个固定seed、超参、最优epoch和参数量。
- `variable_selection.csv`：top50%重训结果；`uncertainty.csv`：与persistence误差差异的时间block-bootstrap区间。
- `interpretation_stability.csv`：5seed训练重要性排名一致性；`data_audit.csv`：实际有效样本数；`paper_comparison.csv`：论文与本次口径并列。
- 两张概括图分别展示预测误差和Tensor代表seed的重要性/时间分布。误差图采用对数纵轴、虚线标出persistence，误差棒是seed均值SE（不是完整泛化置信区间）。代表图固定使用seed11，不根据测试结果选图；所有seed的统计在CSV中。
- 原始数据、所有训练checkpoint/优化器状态、完整逐epoch日志、逐点预测、搜索尝试、独立验证记录和Linux锁依赖都留autoDL。汇总包只包含足以理解结果的文件。

## 3. 论文核心方法如何实现

### 3.1 Full与Tensor的区别

每个变量拥有d维hidden，整体D=N*d。候选更新j_t采用变量独立的input/recurrent映射。IMV-Full的i/f/o门读取所有变量输入和hidden，门可以建模跨变量影响；因此Full不应被描述为完全无跨变量信息。IMV-Tensor的候选、门和memory全部按变量独立更新，最后用mixture attention汇合。对Tensor有扰动独立性测试，对两者有未来输入不影响之前hidden的时序测试。

### 3.2 两级注意力和概率混合

先对每变量前T-1个hidden做归一化时间attention，得到context，再与最后hidden拼接。每变量有独立均值与方差头；变量打分函数共享，softmax为先验pi。混合分布p(y|X)=sum_n pi_n*N(y;mu_n,sigma_n²)。预测严格使用sum(pi*mu)，推断没有使用真实未来y。sigma采用softplus+0.001（训练minmax尺度）的数值下界，防止高斯分量方差塌缩；该数值下界是明确登记的实现选择。

### 3.3 EM、后验重要性与MSE消融

训练E步q_n=softmax(log pi_n+log Gaussian(y;mu_n,sigma_n))，q被冻结后对-logjoint的q加权期望做梯度下降。根据论文式11，I=训练实例q平均；式12的时间重要性为训练alpha平均。式9第三项在E步q固定时只决定全局I的闭式更新，不应凭空额外向theta传梯度。自动测试验证EM与直接混合NLL的梯度一致，以及手算高斯后验数值。

逐epoch重要性轨迹是该epoch在线训练batch的q平均，参数会随batch更新；最终I/T则在**验证集选中的最佳checkpoint、eval模式、完整训练集**上重新准确汇总。两者用途不同。beta/先验均值单独保存，绝不把beta冒称后验I。MSE版本只使用相同结构的点预测MSE，sigma不由MSE训练，因此不对其导出伪高斯后验；MSE的排名是attention，不是式11。

### 3.4 基线实现与适用限制

LSTM是普通单层LSTM回归；DUAL实现input-attention encoder及temporal-attention decoder，decoder另接过去已观测目标，encoder排除目标列；RETAIN用reverse-time双GRU产生时间与embedding维attention后做回归，变量排名取绝对预测贡献。两者是本文从原方法结构独立编写的回归适配，不能冒称原作者基线二进制的完全相同行为。

RF/XGT/ENET把整个过去窗口展平；STRX为有外生回归的local linear trend状态空间；ARIMAX阶数(p,d,q)=(T,0,T)，使用最后已观测时刻的外生变量，预测逐步吸收此前已出现的真实观测。统计模型在有效窗口缺口保持原始时间网格，缺y不回填，缺口exog只用训练均值填充；常数/共线exog仅以训练矩阵QR剔除。统计初次优化先100迭代、未收敛则继续300迭代。失败项保留原始目录，额外训练集数值诊断使用QR预条件和不同求解器；候选必须同时满足优化器收敛、有限归一化优化坐标score最大值≤1e-6、训练似然重放一致、QR预测映射及未来标签隔离检查。修复在独立sidecar目录，登记来源后由验证器和报告统一解析，不覆盖原失败。ARIMAX如采用exact diffuse初始化，属于明确披露的初始化变体，不能称与原approximate diffuse目标完全相同。所有测试预测用原尺度参数和训练固定方差的未剖面化模型过滤，测试数据不估计方差。

本次在实验后对六个状态空间组统一升级了数值验收，不能称为预登记标准。原SML STRX、NASDAQ STRX/ARIMAX虽然有优化器成功flag，在统一QR/logvariance/创新SD坐标下重算的最大归一化score分别约0.04863、0.003285、0.05752，因此另行训练集续优化；这不证明原预测必然错误。原SML ARIMAX与PM STRX未收敛、PM ARIMAX发生非有限数值，均保留原证据。六组最终均以物理参数独立过滤、训练似然与预测映射校验，以及当前/未来标签扰动的因果前缀检查验收。特定原成功组的替代仅由固定原证书和版本2策略授权，不能任意覆盖成功结果。

数值修复依次遇到低内存backend shape错误、profile与原approximate-diffuse训练似然不等价、梯度方向不一致及优化器precision-loss，均作为失败保留，未当成功。最终采用未剖面化似然、可逆QR、log variance和训练创新SD缩放；ARMA保持窗口对应的完整阶数，不因拟合困难降阶。最终exact-diffuse初始化是披露的变体。

PM ARIMAX最终BFGS成功且最大score约4.46e-8，但单一步长1e-5的中心差分存在二阶截断误差。固定同一物理参数后，使用预先固定的12步长网格1e-3至3e-9、三个固定方向种子11/23/47；要求三个相邻步长同时与complex-step方向导数一致（rtol=0.02、atol=2e-6），且三者极差≤2e-6。选第一个满足条件的平台，不挑单个幸运步长。六组最终统一采用同一多步长审计；score≤1e-6和边界KKT≤1e-6门槛保持不变。该认证不优化参数，也不读测试表现决定是否接受。

{table(pd.DataFrame(statistical_rows))}

## 4. 数据、目标、时间切分与泄漏防护

{table(data)}

**SML2010**：公开两段T15文件共4137行；房间温度为目标，16个外生变量加目标历史共17维。按原始行3200/3600切分；窗口10，跨文件和非连续15分钟窗口不使用。正文称每分钟，但图4与文件实际为15分钟，本次遵从可核实源文件。第三方notebook的餐厅温度目标不是本次目标。

**PM2.5**：北京2010—2014逐小时数据，原始43824行。目标pm2.5；DEWP、TEMP、PRES、Iws、Is、Ir，加风向NE/NW/SE/cv四固定哑变量及目标历史，共11维。任何输入或目标缺失的整窗排除，不用未来回填。60/20/20按原始行时间切分来自已登记的公开实现协议，论文未充分指定该数据划分；论文图中只有7维而文字提风向，本次保留风向并显式披露差异。有效样本总数未必等于原文41700，不能强行修改成相同数目。

**NASDAQ**：81只股票与NDX共82维，目标NDX；40560行，原始行35100/37830为切点，窗口10。公开文件已padding，本次不再插值；没有原始逐股票时间戳，无法进一步还原作者padding之前的缺失过程。

**PLANT**：没有取得可确认对应20842序列及9天气变量的原始数据快照。原作者config列的是本地dat路径，推荐PyTorch仓库也未发布它。已查看作者仓库、原研究作者托管论文和大学记录，未将其他光伏数据替代。本项未完成，是全论文复现的明确缺口。

窗口目标为下一时点；train/val/test按目标原始行互不重叠，val/test可使用此前已观测历史（rolling one-step）。这不是在同一个起点开放式预测整个测试区间。x/y minmax只用训练窗口/训练目标拟合；常数列分母1；测试值可超出[0,1]，不会反过来扩大训练尺度。输入目标列的scale与输出y scale分别保存，persistence由输入目标列正确逆变换，避免混用边界略不同的scaler。

## 5. 训练、超参选择、重复与恢复

5个种子固定为11、23、47、71、101。Adam batch64，梯度范数上限5，模型heads dropout，L2只作用于矩阵参数，学习率恒定。训练用MSE或论文EM期望目标；**超参数和checkpoint一律根据验证RMSE**选择，不能根据测试likelihood或测试RMSE选择。

主模型每数据/结构搜索16组预登记均衡组合，覆盖d={{10,15,20,25}}、lr={{0.0005,0.001,0.005,0.01,0.05}}、dropout={{0,0.2,0.5}}与L2={{0.0001,0.001,0.01,0.1,1}}。原论文全部笛卡尔积300组/组合，本次16组是明确的资源折中，**没有做完整穷举**。深度基线6组覆盖h16/32/64/128/256/512，但没有穷举全部学习率/正则。所有搜索seed11、最多80epochs、patience20；正式重复最多150epochs、patience35。MSE消融复用EM选择的架构/训练超参，用于控制结构差异，不能解释为MSE独立最优超参。

{table(pd.DataFrame(search))}

RF搜索depth3/6/10和tree2/50/100/200；XGT搜索depth3/6/10、tree50/100/200及论文6种L2；ENET搜索L1/L2各8个值。RF/XGT用5个固定seed分别拟合；本协议XGT的subsample/colsample默认为1，五seed预测完全一致，SD=0反映该配置的确定性，不能当成五份独立测试误差样本或无泛化不确定性；确定性ENET/STRX/ARIMAX只真实拟合一次，不能复制5行假装独立重复。STRX/ARIMAX的最大迭代与收敛信息单独保存。

每epoch原子保存model、Adam状态、Python/NumPy/torch/CUDA随机状态、epoch、最佳验证轮次与完整历史。续跑严格检查配置、数据和模型/engine源码hash，发生变化时拒绝混合不同实现。回归测试验证“连续3epoch”与“1epoch中断后恢复到3epoch”权重逐元素完全一致。长任务独立后台进程运行，SSH断连不会杀死训练；同一suite有文件锁防止重复启动。GPU计算有其他任务时等待，仅使用已有2080Ti，不购买算力。

## 6. 主要预测结果

原单位RMSE/MAE和训练minmax尺度分别给出。SD是5seed样本标准差(ddof=1)，SE=SD/sqrt(5)；确定性单次拟合的SD/SE在CSV记0，表示未测量训练随机性，而不是已证明无不确定性。不能把SE当作时间序列泛化误差的完整置信区间。

{table(display)}

与persistence差异的额外时间不确定性使用1000次circular moving-block bootstrap，块长20个有效预测窗口，5seed误差先在同一时间点平均，再对时间块抽样。delta MSE = IMV平方误差均值 - persistence平方误差；负数表示IMV更好，95%区间跨0则证据不明确。PM缺口场景中块长按有效窗口计，不代表严格连续20小时；这是该bootstrap的局限。未做多重比较校正，不把它作为所有比较的显著性保证。

{table(uncertain)}

### 6.1 泛化差异、训练预算与负结果诊断

{table(pd.DataFrame(generalization_rows))}

NASDAQ测试目标约97.40%超出训练目标minmax范围（测试缩放值约0.974—1.194）。Full五seed的平均残差均为负，单seed平均偏差约-14.1至-31.3指数点；Tensor约-99.8至-102.7点。这里观察到明显的范围外外推和系统低估，支持“本协议下泛化失败存在”的结论，但不能仅凭它证明失败的唯一原因。SML和PM测试目标均在训练目标范围内，Tensor仍落后，说明目标越界不足以解释所有差距；目标范围内也不保证输入联合分布完全相同。

PM Full五seed都达到150轮预算，最佳验证轮次在140—148附近；本次没有继续增加轮数后挑更好测试结果。深度run的completed/all_runs_completed表示固定训练预算已执行并核验，**不等于神经网络梯度达到数值收敛**。不能把150轮完成与统计模型score收敛混为一谈。有限超参预算、EM似然目标与点预测RMSE的差别、正则/方差下界及数据协议都可能影响结果，当前证据不能分解它们的独立因果贡献；这也不证明论文方法在所有协议下无效。

与persistence的block-bootstrap区间中，仅PM Full的delta MSE区间完全小于0（约[-60.19,-27.88]）；其他五项完全大于0，方向与主要结果一致。这是固定块长和当前评估集下的描述性不确定性结果，不能补偿数据协议不一致或未穷举搜索。

## 7. 变量选择实验与解释证据

每模型、每seed使用该seed全变量模型的**训练集**排名选ceil(N/2)个变量，从头重训，继承全变量验证选定超参，再用新验证最优epoch。IMV按EM后验I与训练Pearson各做一组；Pearson为各变量各lag对下一步训练y的绝对相关系数之均值，常数列相关0。这个lag聚合选择是明确登记的实现细节，原文没有完全指定。DUAL按encoder attention，RETAIN按绝对贡献排名。DUAL的目标历史还提供给decoder，即使未入encoder top50%；因此它的有效输入包括额外目标历史，必须在比较时注明，不能声称所有模型信息预算严格相等。

{table(selection[['dataset','model','objective','selection','n_runs','rmse_raw_mean','mae_raw_mean','rmse_scaled_mean']])}

{chr(10).join(selection_findings)}

六个“数据集×IMV结构”组合中，Pearson筛选的平均RMSE全部低于后验重要性筛选；PM Full后验筛选还使RMSE从约21.13升至31.84。因此本次不支持“后验排名稳定优于简单相关筛选”的结论。Tensor筛选后虽然相对自身全变量有所改善，仍未超过persistence。

如果top50%后误差接近或更低，可支持“本协议下所选变量保留有用预测信息”的有限结论；如果更差，应承认筛选损失，不能笼统重复论文“自动得到更优结果”。Pearson的线性关联和后验的模型内分配不是同一量，负相关取绝对值也不代表负贡献被忽略。此实验不鉴定真实因果变量。

{table(stable)}

{chr(10).join(stability_findings)}

代表Tensor图中目标自身历史（房间温度、pm2.5、NDX）排名较高，显示模型确实将后验质量分配给目标历史；然而Tensor预测仍较差，说明“解释看起来合理”不能替代预测误差验证。热图每套数据使用各自色标，不应直接跨面板按颜色深浅比较绝对差异。

Spearman为5seed两两排名相关的均值，衡量解释稳定性。高预测精度不保证跨seed排名一致；低一致性意味着对单一变量排名的解释应保守。代表性时间图展示训练alpha平均：相对最后观测hidden的lag9..1，对预测目标实际lag10..2；最后观测hidden单独直接进入输出。更集中到最近历史表示该模型在该变量上相对偏向近期状态，不等于真实物理系统只存在这一滞后。

![预测误差](figures/forecast_errors.png)

![变量与时间重要性](figures/importance_temporal.png)

## 8. 参数量、效率与复杂度

原论文Lemma3.1对recurrent部分的单bias参数计数为：标准LSTM 4D²+4ND+4D；Full比它少(N-1)D+(1-1/N)D²；Tensor比它少4(N-1)D+4(1-1/N)D²。N3,d2,D6的手算测试给出Full204、Tensor96、标准240。网络总参数还包括attention、均值与方差头，不能直接拿总参数套Lemma。PyTorch标准LSTM还有两套bias，其实际参数会比上述单bias公式多4D。

真实每轮耗时在各run history中，训练总耗时与参数量在seed_metrics.csv。GPU kernel、Python循环、batch、数据量和收敛epoch会改变实际运行时间，理论O(D²/N+D)并不保证端到端恰好N倍加速。不同模型选择的d/h也可能不同，因此应结合具体参数量与训练时长解释，不能只按理论阶数宣布跑得更快。状态空间sidecar的seconds主要包含相应末次优化/认证和重放，不包括全部早期失败诊断与暖启动计算，因此不能用这个字段给统计方法做端到端效率排名。

## 9. 与论文数值的对应及未达项

{table(paper)}

**directly_comparable=False的原因**：论文没有充分公开相同缩放/PM缺失处理/PLANT数据；SML/NASDAQ表中小数不能在口径未确认时强行当成原单位。本次双口径结果用于透明展示，不通过挑某个归一化因子让数字看起来接近。Full/Tensor方法结构、后验/时间解释、公开数据时间切分和重复比较是本次主要可验证复现内容；论文全部4数据、全超参穷举、原TensorFlow训练逐行运行与原始全部baseline实现未完全完成。

未收敛的经典模型组数：{len(nonconverged)}。若大于0，相关行statistical_converged=False，结果可作求解器诊断，不能当成收敛基准。数值失败的搜索候选数：{len(failures)}；这些候选保留错误与配置、被排除于验证候选集合，未静默更改其超参。它们不是正式重复中的被挑除seed。

{table(pd.DataFrame(failures)) if failures else '本次没有因非有限数值被排除的搜索候选。'}

## 10. 独立验证做了什么

所有195个正式深度run逐一重载最佳checkpoint（先在CPU完成一部分，再在空闲GPU完成余项；每行均绑定同一核验函数、模型源码、数据和artifact SHA，迁移不重训、不清空已验证证据），用与训练导出不同batch大小256独立前向，核对全部测试target_row、预测、原/缩放RMSE与MAE、best epoch是否为验证历史最小值，校验配置/数据/文件SHA。EM的训练后验I还在验证器里按logjoint重新计算，不仅信任导出的importance.csv。15组经典模型逐点预测CSV与原/缩放指标复算。pandas对float32数组导出最短十进制后默认以float64读回；验证器恢复XGT/ENET原预测float32精度、使用处理文件中canonical标签重算，核对原精度的逆变换往返，指标门槛仍为rtol1e-9/atol1e-10。对全部经典组和种子，重建后的指标与原结果差异最大约1.42e-14；没有改动已保存预测/指标或放宽验收。其参数与求解器诊断另存。

验证同时要求完整expected dataset×model×seed×selection集合存在，缺任一正式运行会失败，不能“只核验已完成部分然后称全通过”。搜索运行不允许含测试指标/预测文件；选择配置必须是公开记录验证分数的最小值。数学测试覆盖高斯后验手算、EM/NLL梯度、batch1、归一化、有限梯度、Tensor变量流隔离、因果前缀、参数量、基线与恢复一致性。验证状态：{verified['status']}，深度run {verified['deep_runs']}、经典组 {verified['classic_groups']}。新增断点恢复、审批身份、来源变更、候选收敛与多步长认证的反例测试全部通过，最近完整科学测试为77项；最终发布前还会再次检查。

局限：模型推断核验仍加载同一模型实现，因此数学/数据对齐测试是独立补充；缺少作者原始预处理dat时，无法证明逐样本与作者内部数据一致。没有把自校验通过等同于论文数值完全相同。

## 11. 复现、资源与文件追溯

所有实验只使用用户已有autoDL实例2080Ti与独立安装的开源包；无额外付费API、数据购买、扩容或订阅修改。普通训练CPU线程2，后期统计续优化每进程1线程，GPU单worker，最多60%显存，其他GPU任务出现时等待。项目环境/数据/cache/checkpoint/原始日志完全独立，不复用其他实验文件。

来源：[正式论文与补充材料](https://proceedings.mlr.press/v97/guo19b.html)、[原作者仓库](https://github.com/weilai0980/IMV-LSTM)、[公开数据快照来源](https://github.com/KurochkinAlexey/IMV_LSTM)。固定版本、29项资产SHA与许可边界见仓库sources.lock.json和docs/SOURCE_AUDIT.md。

公开仓库： https://github.com/VictoriaSU551/DSC6013-IMV-LSTM 。原始起点commit89433d8965c8957bd66598b78a83fd06375ba6db；{publication_note}

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements-autodl.txt
venv/bin/python code/setup_project.py
venv/bin/python code/prepare_data.py
PYTHONPATH=code CUBLAS_WORKSPACE_CONFIG=:4096:8 venv/bin/python -m pytest tests -q
PYTHONPATH=code CUBLAS_WORKSPACE_CONFIG=:4096:8 venv/bin/python -m repro.suite
PYTHONPATH=code venv/bin/python -m repro.verify --deep-audit output/reproduction/deep_audit_<真实编号>.json
PYTHONPATH=code venv/bin/python -m repro.report
```

上面是基础训练命令；本次严格统计恢复的实际命令、证书和版本化sidecar流程见仓库docs/REPAIR_NEXT.md与RUNNING_AUTODL.md。固定审批证书只适用于本次原始结果身份；全新数据/运行必须独立生成审计证据与新版本策略，不允许直接改证书或复用旧审批来绕过验收。未下载原始数据时先按来源审计文档取得公开数据并核对SHA。GPU版torch安装应使用匹配驱动的官方wheel；本次Linux完整freeze另保留在远端。不同CUDA/PyTorch/CPU架构即使固定seed也未必逐bit相同，因此以科学指标和公差核验为准。续跑同一结果目录不会覆盖已有证据；实现变更必须新建run集合并在执行记录说明。

## 12. 可以与不可以从本次结果得出的结论

可以评价本次公开数据协议下Full/Tensor与persistence、LSTM、DUAL、RETAIN及传统方法的一步预测误差；可以检查5seed稳定性、后验与先验差异、时间attention形态以及top50%变量保留效果；可以复算已交付指标并在远端重载模型。结论的适用对象是这些特定目标、时间划分和超参预算。

不能从本次结果保证其他市场/年份/任务的优势，不能把后验/注意力解释成因果，不能声称取得未发布的PLANT快照，不能说16组搜索等于300组穷举，不能以口径不明的论文数字给本次原单位误差贴“完全一致”的标签。结果若不支持原论文某一优势，应在主要发现中如实承认，这是严谨复现的一部分。
'''
    (DELIVERY/'复现实验报告.md').write_text(report,encoding='utf-8')
    manifest=[]
    for p in sorted(DELIVERY.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS.txt': manifest.append(sha(p)+'  '+str(p.relative_to(DELIVERY)))
    (DELIVERY/'SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
    atomic_json(OUT/'report_evidence.json',{'status':'generated_requires_human_style_agent_review',
                'files':len(manifest)+1,'bytes':sum(p.stat().st_size for p in DELIVERY.rglob('*') if p.is_file()),
                'report_sha256':sha(DELIVERY/'复现实验报告.md')})
    print('REPORT_PACKAGE_GENERATED',str(DELIVERY))

if __name__=='__main__': main()
