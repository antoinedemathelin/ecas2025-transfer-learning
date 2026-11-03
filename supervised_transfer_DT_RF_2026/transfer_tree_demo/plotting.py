"""Plotting only. No model fitting, selection or scoring is hidden here."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from matplotlib.lines import Line2D
import textwrap
from .distribution import covariance_from_parameters
from .scenarios import bayes_probability

from io import BytesIO
from base64 import b64encode
from IPython.display import HTML, display
    
from matplotlib.colors import LinearSegmentedColormap

COLORS = ('#2563a6','#ec8c36')

SOFT_CMAP = LinearSegmentedColormap.from_list(
    "class_probabilities",
    COLORS,
    N=256
)



def _axes_limits(*arrays):
    X = np.vstack(arrays)
    mins, maxs = np.quantile(X,[0.004, 0.996],axis=0)
    pad = 0.08*(maxs-mins)
    return mins-pad, maxs+pad

def _frame_grid(limits, resolution=190):
    lo,hi=limits
    gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],resolution),np.linspace(lo[1],hi[1],resolution))
    return gx,gy,np.c_[gx.ravel(),gy.ravel()]

def _draw_points(ax, X,y, size=7, max_points=700):
    idx=np.linspace(0,len(X)-1,num=min(len(X),max_points),dtype=int)
    for k,color in enumerate(COLORS):
        sub=idx[y[idx]==k]
        ax.scatter(X[sub,0],X[sub,1],s=size,alpha=.52,c=color,edgecolors='none',label=f'Class {k}')

def _draw_ellipses(ax,spec):
    for k,(w,mu,pc) in enumerate(zip(spec['weights'],spec['means'],spec['cov_params'])):
        if w<1e-9:continue
        cov=covariance_from_parameters(*pc)
        lam,V=np.linalg.eigh(cov)
        idx=lam.argsort()[::-1];lam=lam[idx];V=V[:,idx]
        theta=np.degrees(np.arctan2(V[1,0],V[0,0]))
        ax.add_patch(Ellipse(mu,2*1.4*np.sqrt(lam[0]),2*1.4*np.sqrt(lam[1]),angle=theta,
                             fill=False,lw=1.2,edgecolor='black',alpha=.7))
        ax.scatter(*mu, marker='x',s=42,c='black',linewidth=1.5)
        ax.annotate(f'C{k}',(mu[0]+0.10,mu[1]+0.12),fontsize=8,weight='bold')

def plot_domains(Xs,ys,Xt,yt,source_spec,target_spec,scenario,notes,save=None):
    lim=_axes_limits(Xs,Xt)
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),sharex=True,sharey=True)
    fig.subplots_adjust(top=.82, bottom=.255, wspace=.12)
    for ax,X,y,spec,title in zip(axes,(Xs,Xt),(ys,yt),(source_spec,target_spec),('Source','Target')):
        _draw_points(ax,X,y)
        _draw_ellipses(ax,spec)
        gx,gy,grid=_frame_grid(lim)
        ps=bayes_probability(spec,grid).reshape(gx.shape)
        ax.contour(gx,gy,ps,levels=[.5],colors='black',linewidths=2)
        ax.set_xlim(*[lim[0][0],lim[1][0]]);ax.set_ylim(*[lim[0][1],lim[1][1]])
        ax.set_title(f'{title} — true Bayes frontier (black)')
        ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.grid(alpha=.12)
    fig.suptitle(scenario.replace('_', ' ').upper(), fontsize=13, y=.965)
    text = '\n'.join(textwrap.wrap(' | '.join(notes), width=120))
    fig.text(.5,.035,text,ha='center',va='bottom',fontsize=9)
    if save: fig.savefig(save,dpi=140,bbox_inches='tight')
    plt.show();plt.close(fig)

def plot_predictions(models, Xs,ys,Xt,yt,source_spec,target_spec,scenario,save=None,soft=False):
    """Bayes plus every provided model evaluated on BOTH domains.
    
    """
    names=['Bayes']+list(models)
    n=len(names)
    lim=_axes_limits(Xs,Xt)
    gx,gy,grid=_frame_grid(lim,resolution=165)
    fig,axes=plt.subplots(2,n,figsize=(3.6*n,7.0),sharex=True,sharey=True,squeeze=False,constrained_layout=True)
    for r,(domain,X,y,spec) in enumerate([('Source',Xs,ys,source_spec),('Target',Xt,yt,target_spec)]):
        bayes=bayes_probability(spec,grid).reshape(gx.shape)
        yt_bayes=(bayes_probability(spec,X)>=.5).astype(int)
        for j,name in enumerate(names):
            ax=axes[r,j]
            if name=='Bayes':
                zz=bayes
                score=accuracy_score(y,yt_bayes)
            else:
                m=models[name]
                if hasattr(m,'predict_proba'):
                    zz=m.predict_proba(grid)[:,1].reshape(gx.shape)
                else:
                    zz=np.asarray(m.predict(grid),dtype=float).reshape(gx.shape)
                score=accuracy_score(y,m.predict(X))
                
            
            #ax.contourf(gx,gy,zz,levels=[-.01,.5,1.01],colors=COLORS,alpha=.18)
            if soft:
                ax.contourf(
                    gx, gy, zz,
                    levels=np.linspace(0, 1, 101),
                    cmap=SOFT_CMAP,
                    vmin=0,
                    vmax=1,
                    alpha=0.40
                )
            else:
                ax.contourf(
                    gx, gy, zz,
                    levels=[-.01, .5, 1.01],
                    colors=COLORS,
                    alpha=0.18
                )
            ax.contour(gx,gy,zz,levels=[.5],colors='#a43336',linewidths=1.2,linestyles='dashed')

            ax.contour(gx,gy,bayes,levels=[.5],colors='black',linewidths=1.6)
            _draw_points(ax,X,y,size=5,max_points=450)
            ax.set_title(f'{domain}: {name}\naccuracy = {score:.3f}',fontsize=10)
            ax.set_xlim(lim[0][0],lim[1][0]);ax.set_ylim(lim[0][1],lim[1][1]);ax.grid(alpha=.07)
            if r==1:ax.set_xlabel('$x_1$')
            if j==0:ax.set_ylabel('$x_2$')
    fig.suptitle(f'{scenario}: black = true Bayes, dashed red = model 0.5 frontier',fontsize=11)
    if save:fig.savefig(save,dpi=130,bbox_inches='tight')
    plt.show();plt.close(fig)

def plot_selection_frequencies(index_arrays,algorithms,title,save=None):
    """Frequency of STRF tree-level algorithm choices, with seed uncertainty."""
    counts=[]
    for indexes in index_arrays:
        ids=np.asarray(indexes,dtype=int)
        if np.any((ids<0)|(ids>=len(algorithms))):
            raise ValueError('Invalid selector strategy indexes')
        counts.append(np.bincount(ids,minlength=len(algorithms))/len(ids))
    count=np.array(counts)
    mean=count.mean(axis=0)
    std=count.std(axis=0,ddof=1) if len(count)>1 else np.zeros_like(mean)
    fig,ax=plt.subplots(figsize=(8,3.6))
    ax.bar(algorithms,mean,yerr=std,capsize=4,color=['#64748b','#d1a74b','#7aa2c2','#62a893','#c88888'][:len(algorithms)])
    ax.set_ylim(0,min(1.0,max(.30,float(np.max(mean+std)*1.16))))
    ax.set_ylabel('Share of forest trees')
    ax.set_title(title+f' (n={len(count)} repetitions)')
    ax.grid(axis='y',alpha=.15)
    if save:fig.savefig(save,dpi=150,bbox_inches='tight')
    plt.show();plt.close(fig)
    return mean

def _node_map(estimator):
    """Map L/R decision paths to nodes (not arbitrary sklearn node indices)."""
    tree = estimator.tree_
    mapping = {}
    classes = np.asarray(estimator.classes_)

    def traverse(node, path):
        leaf = tree.feature[node] < 0
        counts = np.asarray(tree.value[node]).reshape(-1)
        majority = classes[int(np.argmax(counts))]
        mapping[path] = {
            'id': int(node), 'leaf': bool(leaf),
            'feature': None if leaf else int(tree.feature[node]),
            'threshold': None if leaf else float(tree.threshold[node]),
            'majority': int(majority),
        }
        if not leaf:
            traverse(int(tree.children_left[node]), path + 'L')
            traverse(int(tree.children_right[node]), path + 'R')
    traverse(0, '')
    return mapping


def _node_changes(original, transferred, threshold_tol=1e-3):
    """Flag structural edits, threshold-only edits and leaf relabels.

    Recursively compare shared L/R paths only until an incompatible split occurs;
    descendants after a split-feature change cannot be meaningfully matched.
    """
    a = _node_map(original)
    b = _node_map(transferred)
    kinds = {}

    def compare(path):
        if path not in a or path not in b:
            kinds[path] = 'structural'
            return
        old, new = a[path], b[path]
        if old['leaf'] != new['leaf']:
            kinds[path] = 'structural'
            return
        if old['leaf']:
            if old['majority'] != new['majority']:
                kinds[path] = 'relabel'
            return
        if old['feature'] != new['feature']:
            kinds[path] = 'structural'
            return
        if abs(old['threshold'] - new['threshold']) > threshold_tol:
            kinds[path] = 'threshold'
        compare(path + 'L')
        compare(path + 'R')

    compare('')
    return kinds


def plot_tree_diff(original, transferred, title='Source → transferred tree',
                   max_depth=4, threshold_tol=1e-3, save=None):
    """Two aligned tree diagrams, class-colored nodes and distinct edit borders.

    Blue/orange fill: majority class in EACH node (internal and terminal).
    Solid red: structural change (pruning, expansion, or split feature).
    Dashed red: changed threshold, SAME split feature.
    Solid amber: leaf prediction class changed without restructuring.

    Correspondence follows L/R paths up to the first structural mismatch.
    Large trees are clipped at max_depth for readability; deeper edits are
    summarized but not drawn. This is a structural diagnostic, not a proof of
    causal correspondence between re-trained trees.
    """
    kinds = _node_changes(original, transferred, threshold_tol=threshold_tol)
    palettes = {0: '#dcecfb', 1: '#ffe2c6'}

    n_visible_leaves = min(
        2 ** max_depth,
        max(original.get_n_leaves(), transferred.get_n_leaves())
    )
    
    fig_width = max(22, 3.0 * n_visible_leaves)
    fig_height = max(10, 1.8 * (max_depth + 2))
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_width, fig_height)
    )
    
    
    fig.subplots_adjust(top=.86, bottom=.13, left=.025, right=.975, wspace=.12)
    
    
    for ax, model, name in zip(axes,(original,transferred),('Source tree','Transferred tree')):
        mapping = _node_map(model)
        t = model.tree_
        artists = plot_tree(model, ax=ax, max_depth=max_depth,
                            feature_names=['x1','x2'], class_names=['Class 0','Class 1'],
                            filled=False, rounded=True, fontsize=8, impurity=False,
                            proportion=True)

        node_artists = [a for a in artists if a.get_text().strip() not in ('True', 'False')]

        visible = []
        def walk(node, path, depth):
            if depth > max_depth:
                visible.append((None, path))
                return
            visible.append((int(node), path))
            if t.feature[node]>=0:
                walk(int(t.children_left[node]), path+'L', depth+1)
                walk(int(t.children_right[node]), path+'R', depth+1)
        walk(0,'',0)
        if len(node_artists)!=len(visible):
            raise RuntimeError(f'Tree artist mapping mismatch: {len(node_artists)} vs {len(visible)}')
        for artist,(node,path) in zip(node_artists, visible):
            patch = artist.get_bbox_patch()
            if patch is None:
                artist.set_bbox(dict(boxstyle='round,pad=.25',fc='white',ec='#94a3b8'))
                patch=artist.get_bbox_patch()
            if node is None:
                patch.set_facecolor('#f3f4f6')
                patch.set_edgecolor('#94a3b8')
                continue
            majority = mapping[path]['majority']
            patch.set_facecolor(palettes.get(majority, '#eeeeee'))
            kind = kinds.get(path)
            if kind=='structural':
                patch.set_edgecolor('#d01b26');patch.set_linewidth(2.8);patch.set_linestyle('solid')
            elif kind=='threshold':
                patch.set_edgecolor('#d01b26');patch.set_linewidth(2.8);patch.set_linestyle((0,(4,2)))
            elif kind=='relabel':
                patch.set_edgecolor('#b57903');patch.set_linewidth(2.6);patch.set_linestyle('solid')
            else:
                patch.set_edgecolor('#8b95a3');patch.set_linewidth(.8);patch.set_linestyle('solid')
        ax.set_title(f'{name}: {t.node_count} nodes / {t.n_leaves} leaves',fontsize=12)
    legend=[Patch(facecolor=palettes[0],edgecolor='none',label='Majority class 0'),
            Patch(facecolor=palettes[1],edgecolor='none',label='Majority class 1'),
            Line2D([0],[0],color='#d01b26',lw=2.8,label='Structural change (prune / expand / feature)'),
            Line2D([0],[0],color='#d01b26',lw=2.8,ls='--',label='Threshold changed (same feature)'),
            Line2D([0],[0],color='#b57903',lw=2.6,label='Leaf majority relabeled')]
            
    fig.legend(handles=legend,loc='lower center',ncol=3,fontsize=9,frameon=False,bbox_to_anchor=(.5,.04))
    fig.suptitle(f'{title}  |  {len(kinds)} changed matched paths',fontsize=14,y=.97)

    #if save: fig.savefig(save,dpi=145,bbox_inches='tight')
    #plt.show();plt.close(fig)
    #return kinds
        
    if save:
        fig.savefig(save, dpi=145, bbox_inches="tight")
    
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
    
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    width_px = int(fig_width * 115)
    
    display(HTML(
        f'<div style="width:100%; overflow-x:auto;">'
        f'<img src="data:image/png;base64,{encoded}" '
        f'style="width:{width_px}px; max-width:none;">'
        f'</div>'
    ))
    
    plt.close(fig)
    return kinds




def plot_repeated_accuracy(repeated_df, title='Repeated target accuracy', save=None):
    """Show individual fresh-draw scores and mean +/- SD (fitting stays in notebook).

    `repeated_df` must have `rep`, `method` and `target_accuracy` columns.
    Points for the same method represent independent samples from the *same*
    underlying source and target distributions, not changing transformations.
    """
    if repeated_df.empty:
        raise ValueError('No repeated evaluations to plot')
    grouped = repeated_df.groupby('method', sort=False)['target_accuracy']
    stats = grouped.agg(['mean', 'std', 'count']).reset_index()
    methods = list(stats['method'])
    xs = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(8, len(methods)*1.65), 4.6))
    for j, name in enumerate(methods):
        values = repeated_df.loc[repeated_df['method'] == name, 'target_accuracy'].to_numpy()
        jitter = np.linspace(-.10, .10, len(values)) if len(values)>1 else [0.0]
        ax.scatter(j + jitter, values, s=24, alpha=.44, zorder=2, color='#607b94')
    ax.errorbar(xs, stats['mean'], yerr=stats['std'].fillna(0),
                fmt='o', color='#b13030', capsize=5, markersize=7,
                linewidth=2.2, zorder=3, label='Mean ± 1 SD')
    ax.set_xticks(xs, methods, rotation=20, ha='right')
    
    ###
    # Include individual scores and mean ± standard deviation
    std = stats["std"].fillna(0)
    
    ymin = min(
        repeated_df["target_accuracy"].min(),
        (stats["mean"] - std).min()
    )
    ymax = max(
        repeated_df["target_accuracy"].max(),
        (stats["mean"] + std).max()
    )
    
    # Add 10% vertical padding, with a minimum of 0.01
    padding = max(0.01, 0.10 * (ymax - ymin))
    
    ax.set_ylim(
        max(0, ymin - padding),
        min(1, ymax + padding)
    )
    ###
    #ax.set_ylim(0, 1)
    ax.set_ylabel('Held-out target accuracy')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return stats
