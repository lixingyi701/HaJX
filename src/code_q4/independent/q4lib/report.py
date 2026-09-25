import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def markdown_table(df):
    headers=list(df.columns)
    rows=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for values in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(format(v,'.6g') if isinstance(v,float) else str(v) for v in values)+' |')
    return '\n'.join(rows)

def resource_placeholder(out):
    dump(out/'interface/P4_Q3_symbolic_interface.json',{
        'status':'DEFERRED_Q3','numeric_resource_prediction_enabled':False,
        'requires':['frozen P2 loss callable and units/support','P3 quality cost g(Q), N/Q bounds, reference D','Q2-to-C6 paired loss calibration'],
        'objective':'min L_P2(N,D,Q,p_star)',
        'constraint':'(6+eta*Lctx)*N*D + D*max(g(Q)-g(Q0),0) <= C(t); D<=Dmax(t); Q0<=Q<=1',
        'D_star_given_N_Q':'min(C(t)/((6+eta*Lctx)*N+max(g(Q)-g(Q0),0)), Dmax(t))',
        'eta':2e-4,'score_formula':'100*sum_j w_j*m_j(calibrate(L_star(t)))',
        'reason':'Q3 dependency explicitly postponed by user. No zero-filled substitute or obsolete Q2 formula.',
        'double_counting_rule':'Do not add empirical time trend to the full quality gain of the resource chain'})
    csv(out/'interface/P4_resource_scenarios.csv',[dict(status='DEFERRED_Q3',Dmax_scenario=s,bridge_status='BRIDGE_UNCALIBRATED',numeric_result=None) for s in ['unbounded','D_ref','2D_ref']])

def figures(out):
    f=out/'figures';f.mkdir(exist_ok=True)
    flow=pd.read_csv(out/'tables/filter_flow.csv');fig,ax=plt.subplots(figsize=(10,4));ax.barh(flow.stage,flow.n);ax.set_xlabel('Models');fig.tight_layout();fig.savefig(f/'F9_filter_audit.png',dpi=160);plt.close(fig)
    ck=pd.read_csv(out/'tables/C8_C2_replication.csv')
    if len(ck):
        fig,axs=plt.subplots(2,3,figsize=(11,7))
        for dim,ax in zip(DIMS,axs.flat):
            d=ck[ck.dimension==dim];ax.scatter(d.official,d.normalized_clipped_aggregate,s=3,alpha=.25);ax.plot([0,100],[0,100],color='gray');ax.set_title(dim);ax.set_xlabel('C2');ax.set_ylabel('C8 candidate rebuild')
        fig.tight_layout();fig.savefig(f/'F10_score_replication.png',dpi=160);plt.close(fig)
    br=pd.read_csv(out/'tables/bridge_fitted.csv')
    if len(br):
        fig,axs=plt.subplots(2,3,figsize=(11,7))
        for dim,ax in zip(DIMS,axs.flat):
            d=br[br.dimension==dim].sort_values('loss');ax.scatter(d.loss,d.observed,s=10);ax.plot(d.loss,d.prediction,color='orange');ax.set_title(dim);ax.set_xlabel('Reported loss (mixed sources)');ax.set_ylabel('Score')
        fig.suptitle('Exploratory only: cross-source / Q2 calibration unavailable');fig.tight_layout();fig.savefig(f/'F11_bridge_exploratory.png',dpi=160);plt.close(fig)
    ob=pd.read_csv(out/'tables/observed_frontiers.csv');fc=pd.read_csv(out/'interface/P4_frontier_forecast.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for typ,ax in zip(['base','chat'],axs):
        for scope,g in ob[ob.model_type==typ].groupby('scope'):ax.plot(pd.to_datetime(g.date),g.frontier,label=scope)
        if 'central' in fc:
            s=fc[(fc.model_type==typ)&(fc.compute_multiplier==.5)&(fc.technology_multiplier==1)&((fc.tau==.9)|fc.tau.isna())]
            if len(s):ax.plot(pd.to_datetime(s.target),s.central,linestyle='--',label='conditional scenario')
        ax.set_title(typ);ax.set_ylabel('Score');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(f/'F12_frontier.png',dpi=160);plt.close(fig)
    de=pd.read_csv(out/'interface/P4_decomposition.csv')
    if len(de):
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(de));w=.25
        for k,offset in [('scale',-w),('time',0),('residual',w)]:ax.bar(x+offset,de[k],width=w,label=k)
        ax.set_xticks(x,de.model_type);ax.set_ylabel('Score points');ax.legend();fig.tight_layout();fig.savefig(f/'F13_contributions.png',dpi=160);plt.close(fig)

def write_report(out,m,origin,cutoff,cfg):
    dm=json.loads((out/'interface/P4_decomp_models.json').read_text());br=json.loads((out/'interface/P4_bridge_params.json').read_text());post=json.loads((out/'tables/posttrain_status.json').read_text())
    rep=pd.read_csv(out/'tables/C8_replication_summary.csv');forecast=pd.read_csv(out/'interface/P4_frontier_forecast.csv')
    lines=['# 问题四运行报告','',f'- 预测起点：{origin.date()}；数据截止：{cutoff.date()}。',
        f'- 去重、截止后的模型：{len(m)}；严格评分样本：{m.include_primary.sum()}；严格算力面板：{m.include_panel.sum()}。',
        '- “严格”指本实现的证据筛选规则：C8 实际评测或 C4 开放权重证据，加许可元数据允许列表；不等同于许可证全文与每个 revision 已逐一核验。',
        '- 当前快照无法恢复所有历史可用时间；发布日分析属于事后重建。没有把 2025 年附件预测平移到 2026 年。',
        f'- 家族按发布账号保守聚类；bootstrap={cfg["bootstrap"]}，随机种子={cfg["seed"]}。','',
        '## 1. 动力学与贡献','', '模型为 `100 sigmoid(a+b ln(C/Cref)+d(t-tref))`，b≥0，d可正可负。时间项是关联代理，不能称为纯技术因果效应。', '']
    for typ,v in dm.items():lines.append(f'- {typ}：{v.get("status")}；n={v.get("n")}，独立发布账号簇={v.get("families")}，季度={v.get("quarters")}，识别状态={v.get("identification_status","未达到拟合门槛")}。')
    lines+=['','路径积分按季度冠军对应算力的对数线性路径计算；保留负贡献、抵消及未解释差额。模型变化接近零、区间跨零或弱识别时不报占比。',
        '','## 2. 前沿预测','',f'- 状态：{", ".join(forecast.status.dropna().unique())}。',
        '- 样本达到门槛时输出 0.75/0.90/0.95 条件分位；仅满足均值模型门槛时输出条件均值，并明确它不是前沿。',
        '- 算力趋势来自 C4 开放权重语言模型的宽口径代理，许可证覆盖与严格评分面板并不完全相同。趋势不足时必须显式传入假设增速。',
        '- 预测区间只含模型簇拟合及面板算力锚点不确定性；给定增速情景，不含实际最大值分布。',
        '- 附件评测时间跨度不足 12 个月，不能验证 12/24 个月外推准确性；回测状态文件如实记录。',
        '','## 3. 多维评分与复算','',
        '- 主分数核验六维均值；输出六维向量、最低维度、离散度和每维加倍权重的六种敏感性。非线性能力响应通过动力学模型处理，不将非线性响应系数直接平均当贡献。',
        '- C8 保存叶子任务、原始率、未截断/截断基线分；组汇总不重复计数。不同 revision 的任务不会拼接。',
        '- 以下为候选评分规则的复算核验；有偏差即保留 MISMATCH，不能称完全重现。','',markdown_table(rep),
        '','## 4. Loss–Benchmark','',
        '- 高可比层及每来源单独检查跨族支持；混合来源仅输出探索曲线及族外误差。',
        '- Q2→C6 尺度没有配对校准证据，状态 BRIDGE_UNCALIBRATED。局部曲线 bootstrap 不能代替跨来源校准误差。',
        f'- 配对分析状态：{post["status"]}；名称候选 {post["candidate_pairs"]} 对，须核实底座与评测设置后才能用于推断。',
        '','## Q3 依赖','',
        'C 组经验链无需 Q3 启动。资源优化链以公式接口 `P4_Q3_symbolic_interface.json` 保留，状态 DEFERRED_Q3；数值资源前沿未生成。',
        '','## 实现范围与后续补充','',
        '- 已实现：输入审计、身份匹配、评分复算、评分敏感性、可解释响应模型、路径积分、条件预测、桥接、配对接口、保护性降级和复现记录。',
        '- 未实现的完整 Spec 扩展：Q3 数值求解及资源图；经过跨域校准的联合误差传播；有足够实时历史数据后的长周期回测及区间覆盖检验。',
        '- 数值模型门槛未满足不代表需要调低门槛。优先补齐许可核验、精确模型匹配、训练算力范围与独立家族。',
        '- 查看 `tables/pair_candidates.csv` 与模型审计表，填 overrides/pairs 后可重新运行；不自动将候选升级为证据。']
    if (out/'sensitivity_broad/interface/P4_decomp_models.json').exists():
        broad=json.loads((out/'sensitivity_broad/interface/P4_decomp_models.json').read_text())
        lines+=['','## 宽口径敏感性（不替代主结果）','',
            '允许许可或权重证据未完全审查的模型进入敏感性组，仍要求可靠身份匹配。结果位于 sensitivity_broad/；若只达到均值门槛，输出为条件均值，不能称前沿。','']
        for typ,v in broad.items():
            lines.append(f'- {typ}：{v.get("status")}；n={v.get("n")}；发布账号簇={v.get("families")}；基线比较选择={v.get("validation_selected_form","无")}。')
    (out/'Q4_运行报告.md').write_text('\n'.join(lines),encoding='utf-8')
