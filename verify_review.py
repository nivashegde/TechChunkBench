"""Re-verification of reviewer findings (Nov 15). Prints corrected statistics."""
import pandas as pd, numpy as np, json, warnings
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
CORE=["fixed_size","fixed_overlap","structure_aware","semantic"]; M=["hit1","hit3","hit5","mrr","ndcg5","cprec"]
def load(p,pre,unit):
    d=pd.read_csv(p)
    if "error" in d.columns: d=d[d["error"].isna()]
    d=d.drop_duplicates([unit,"strategy","chunk_size","embedding_model"],keep="last").copy()
    d["model"]=d.embedding_model.str.split("/").str[-1]; d["corpus"]=d[unit]
    for m in M: d[m]=d[pre+m]
    return d
e1=load("results/raw/e1_results_final.csv","span_","corpus_id"); e2=load("results/raw/e2v2_results_final.csv","","dataset")
e1c=e1[e1.strategy.isin(CORE)].assign(regime="R2"); e2c=e2[e2.strategy.isin(CORE)].assign(regime="R3")
pool=pd.concat([e1c,e2c],ignore_index=True)
print("== B1 interaction via nested models (SS increment / total SS) ==")
for m in M:
    y=pool[m]; sst=((y-y.mean())**2).sum()
    base=smf.ols(f"{m} ~ C(strategy)+C(chunk_size)+C(model)+C(corpus)",pool).fit()
    reg=smf.ols(f"{m} ~ C(strategy)+C(chunk_size)+C(model)+C(corpus)+C(strategy):C(regime)",pool).fit()
    cor=smf.ols(f"{m} ~ C(strategy)+C(chunk_size)+C(model)+C(corpus)+C(strategy):C(corpus)",pool).fit()
    a1=sm.stats.anova_lm(base,reg); a2=sm.stats.anova_lm(base,cor)
    print(f"{m:6s} strat x regime: eta2={100*(base.ssr-reg.ssr)/sst:.2f}% df={a1.df_diff[1]:.0f} p={a1['Pr(>F)'][1]:.1e} | strat x corpus: eta2={100*(base.ssr-cor.ssr)/sst:.2f}% df={a2.df_diff[1]:.0f} p={a2['Pr(>F)'][1]:.1e} | main strategy eta2={100*sm.stats.anova_lm(base,typ=2).loc['C(strategy)','sum_sq']/sst:.2f}%")
print("\n== S1 within-corpus eta2 (strategy), per corpus, then medians ==")
def within(d,m):
    out={}
    for c,g in d.groupby("corpus"):
        an=sm.stats.anova_lm(smf.ols(f"{m} ~ C(strategy)+C(chunk_size)+C(model)",g).fit(),typ=2); out[c]=an.loc["C(strategy)","sum_sq"]/an["sum_sq"].sum()
    return pd.Series(out)
for m in ["mrr","hit5","ndcg5"]:
    w2=within(e1c,m); w3=within(e2c,m)
    print(f"{m}: R2 median {100*w2.median():.1f}% (IQR {100*w2.quantile(.25):.1f}-{100*w2.quantile(.75):.1f}) | R3 {(100*w3).round(1).to_dict()} median {100*w3.median():.1f}% | Mann-Whitney p={stats.mannwhitneyu(w2,w3).pvalue:.3f}")
# partial eta2 pooled per regime
print("\n== partial eta2 of strategy (SS_s/(SS_s+SS_res)) ==")
for nm,d in [("R2",e1c),("R3",e2c)]:
    r={}
    for m in M:
        an=sm.stats.anova_lm(smf.ols(f"{m} ~ C(strategy)+C(chunk_size)+C(model)+C(corpus)",d).fit(),typ=2); r[m]=round(100*an.loc["C(strategy)","sum_sq"]/(an.loc["C(strategy)","sum_sq"]+an.loc["Residual","sum_sq"]),1)
    print(nm,r)
print("\n== B2 structured files ==")
nf=e2.drop_duplicates("corpus").set_index("corpus").n_files; fr=e2[e2.strategy=="structure_legal"].groupby("corpus").structured_files_frac.mean(); print((fr*nf).round(0).to_dict(), "total", round((fr*nf).sum()), "/", nf.sum(), "=", round((fr*nf).sum()/nf.sum(),3)); print("generic:", (e2[e2.strategy=="structure_aware"].groupby("corpus").structured_files_frac.mean()*nf).round(1).to_dict())
print("\n== B5 dochit5 ==", e2c.groupby("corpus").dochit5.mean().round(2).to_dict())
print("\n== B6 cprec means ==", "R2", e1c.groupby("strategy").cprec.mean().round(3).to_dict(), "R3", e2c.groupby("strategy").cprec.mean().round(3).to_dict())
for nm,d in [("R2",e1c),("R3",e2c)]:
    dd=d[d.strategy!="fixed_overlap"]; an=sm.stats.anova_lm(smf.ols("cprec ~ C(strategy)+C(chunk_size)+C(model)+C(corpus)",dd).fit(),typ=2); print(nm,"cprec eta2 without fixed_overlap:",round(100*an.loc["C(strategy)","sum_sq"]/an["sum_sq"].sum(),1),"pct p=", float(an.loc["C(strategy)","PR(>F)"]))
print("\n== R2: which strategy has best mean MRR per corpus ==", e1c.groupby(["corpus","strategy"]).mrr.mean().unstack().idxmax(axis=1).value_counts().to_dict())
print("== R3 best per sub-benchmark (core) ==", e2c.groupby(["corpus","strategy"]).mrr.mean().unstack().idxmax(axis=1).to_dict())
print("\n== S12 locate rates ==", "R2", e1.groupby("strategy").chunk_locate_rate.mean().round(3).to_dict(), "min semantic", e1[e1.strategy=="semantic"].groupby("corpus").chunk_locate_rate.mean().round(2).nsmallest(3).to_dict(), "| R3", e2.groupby("strategy").chunk_locate_rate.mean().round(3).to_dict())
print("\n== B11 feature correlations (outcome: within-corpus eta2 of strategy on MRR, core strategies) ==")
F=pd.read_csv("results/aggregated/mustdo_D_corpus_features.csv")[["corpus","headings_per_10k","mean_doc_chars","mean_evidence_chars","regime"]]
w=pd.concat([within(e1c,"mrr"),within(e2c,"mrr")]).rename("eta").reset_index().rename(columns={"index":"corpus"})
adv=pd.concat([e1c,e2c]).groupby(["corpus","strategy"]).mrr.mean().unstack(); adv=(adv.structure_aware-adv.fixed_size).rename("adv").reset_index()
h5=pd.concat([e1c,e2c]).groupby("corpus").hit5.mean().rename("h5").reset_index()
F=F.merge(w).merge(adv).merge(h5); F["ld"]=np.log10(F.mean_doc_chars); F["le"]=np.log10(F.mean_evidence_chars)
for c in ["headings_per_10k","ld","le","h5"]:
    r,p=stats.spearmanr(F[c],F.eta); r2,p2=stats.spearmanr(F[c],F.adv); print(f"{c:18s} vs eta2: rho={r:+.2f} p={p:.3f} | vs SA-fixed: rho={r2:+.2f} p={p2:.4f}")
o=smf.ols("eta ~ headings_per_10k + ld + le",F).fit(); print("OLS p(headings)=%.3f adjR2=%.2f"%(o.pvalues["headings_per_10k"],o.rsquared_adj)); o2=smf.ols("eta ~ headings_per_10k + ld + le",F[F.regime=="R2_technical"]).fit(); print("R2-only p(headings)=%.3f"%o2.pvalues["headings_per_10k"])
F2=F[F.regime=="R2_technical"]; r,p=stats.spearmanr(F2.headings_per_10k,F2.eta); print("R2-only rho=%.2f p=%.3f"%(r,p)); print("doc chars: R2 min/max", F2.mean_doc_chars.min(), F2.mean_doc_chars.max(), "R3", F[F.regime=="R3_legal"].set_index("corpus").mean_doc_chars.round(0).to_dict())
print("\n== S2 generation eval paired tests ==")
J=pd.DataFrame([json.loads(l) for l in open("results/gen_eval/judged.jsonl")]).dropna(subset=["correctness"]); P=J.pivot_table(index="qa_id",columns="strategy",values="correctness"); Pm=J.pivot_table(index="qa_id",columns="strategy",values="mrr")
print("Friedman core correctness:", stats.friedmanchisquare(*[P[s] for s in CORE])); print("Friedman core faithfulness:", stats.friedmanchisquare(*[J.pivot_table(index='qa_id',columns='strategy',values='faithfulness')[s] for s in CORE]))
def wl(a,b,T=P):
    d=(T[a]-T[b]); return round(d.mean(),3), round(stats.wilcoxon(d).pvalue,4)
pairs=[(a,b) for i,a in enumerate(CORE) for b in CORE[i+1:]]; ps=[]
for a,b in pairs: r=wl(a,b); ps.append(r[1]); print(" ",a,"-",b,r)
o=np.argsort(ps); adj=[None]*6; run=0
for k,i in enumerate(o): run=max(run,ps[i]*(6-k)); adj[i]=round(min(run,1),3)
print(" Holm:",dict(zip([f"{a}-{b}" for a,b in pairs],adj)))
print(" boundary (noprefix-fixed):",wl("structure_noprefix","fixed_size")," prefix (sa-noprefix):",wl("structure_aware","structure_noprefix")," MRR sa-fixed in subsample:",wl("structure_aware","fixed_size",Pm))
Jc=J[J.strategy.isin(CORE)]; print(" r (core rows):", round(stats.pearsonr(Jc.mrr,Jc.correctness)[0],2), "| per-strategy r:", {s: round(stats.pearsonr(J[J.strategy==s].mrr,J[J.strategy==s].correctness)[0],2) for s in CORE})
