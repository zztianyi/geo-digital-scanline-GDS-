"""Branch-only gallery: FaceTrack membership, never main-route membership."""
from pathlib import Path
import json,pickle
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from diagnose_missing_routes import latest,SOURCE
from validate_physical_topology import csv_rows

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
GREEN='#187f68'


def crop(segment,bounds):
    p=np.asarray(segment,dtype=float);a,b=p;d=b-a;t0,t1=0.,1.
    for axis,lo,hi in [(0,bounds[0],bounds[1]),(1,bounds[2],bounds[3])]:
        if abs(d[axis])<1e-15:
            if not lo-1e-10<=a[axis]<=hi+1e-10:return None
        else:
            u,v=sorted(((lo-a[axis])/d[axis],(hi-a[axis])/d[axis]));t0=max(t0,u);t1=min(t1,v)
    return np.array([a+t0*d,a+t1*d]) if t1-t0>1e-12 else None


def panel(out,r,tid,rows,bounds,detail=False):
    keys=r['target_keys'];fig,axs=plt.subplots(1,5,figsize=(18.5,7),sharex=True,sharey=True)
    for ax,key in zip(axs,keys):
        members=[e for e in rows if e['s']==key];pieces=[crop(e['points_uz'],bounds) for e in members]
        pieces=[p for p in pieces if p is not None]
        if pieces:ax.add_collection(LineCollection(pieces,colors=GREEN,lw=1.65))
        else:ax.text(.5,.5,'本窗口无该来源分支',ha='center',va='center',transform=ax.transAxes,fontsize=10)
        bids=sorted({e['branch_id'] for e in members if crop(e['points_uz'],bounds) is not None})
        ax.set_title(f's = {float(key):.2f} m\n分支 '+(', '.join('B'+str(b) for b in bids) or '无'))
        ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.18);ax.set_xlabel('径向偏移 u（m）')
        ax.ticklabel_format(useOffset=False,style='plain')
    axs[0].set_ylabel('高程 z（m）')
    fig.suptitle(f'{tid}｜相同面来源筛选后的五条邻线分支'+('｜细节放大' if detail else ''),fontsize=17)
    fig.text(.5,.083,'绿色：仅属于本 FaceTrack 的原始分支；五栏使用相同坐标。分支编号可不同，多段保留，不补线、不平滑。',ha='center',fontsize=11)
    fig.text(.5,.038,'本图未使用最终主轨的选择结果，也没有按相似程度挑选曲线；一致性由人工审核。',ha='center',fontsize=11)
    fig.subplots_adjust(left=.066,right=.985,top=.80,bottom=.20,wspace=.11)
    filename=f'{tid}'+('_detail' if detail else '')+'.png';fig.savefig(out/'figures'/filename,dpi=160);plt.close(fig)
    return filename


def main(out):
    records=[];groups=[]
    # Windows are explicitly chosen from prior visible details, not a score threshold.
    zoom={'C':(1361.8,1364.15),'D':(1410.8,1413.1),'E':(1384.6,1387.),'F':(1392.8,1395.),'I':(1378.7,1381.1),'J':(1401.6,1403.)}
    descriptions={
      'A':'0.85：观察同来源内是否有形态渐变或突然增减的细节。',
      'B':'78.40：同一来源的长分支对照。',
      'C':'103.50–103.70：重点看 1362–1364 m 的折返随 s 的变化。',
      'D':'122.15–122.35：两个来源分别比较，不混在一起选曲线。',
      'E':'104.20–104.40：三个来源分别看尖折、平台和回折是否对应。',
      'F':'130.35–130.55：两张几乎重合的面各自形成一组，观察每组的细节连续性。',
      'G':'122.85–123.05：直接展示所有同来源分支，不受主轨是否缺失影响。',
      'H':'116.10–116.30：直接检查平台/折返在五条分支上是否对应。',
      'I':'100.05–100.25：四个来源分别观察，有缺席也如实显示。',
      'J':'108.70–108.90：重点看 1402 m 附近的小回环及两端细节。',
      'K':'1.85–2.05：单一来源对照；窗口来自上一轮位置，不代表已定位原锚点平局。',
      'L':'67.50–67.70：多来源与短小片段；窗口来自上一轮边界代理位置。'}
    blocks=['# 同一 FaceTrack 下的相邻分支：人工审核图册',
      '本册按你的要求只验证“FaceTrack 匹配后，分支细节是否一致”。不把当前主轨的保留结果作为过滤条件，不计算 TC 通过分数，不自动判定一致或错误。',
      '使用上一轮已经建立的局部真实共享边 FaceTrack。每组只显示一个 FaceTrack，按相邻五条纵向测线排列；全体该来源的原始分支片段都保留。若同一来源在某切面里仍有多条/多段，不擅自选其中一条。',
      '读图时依次查看：平台与尖折是否对应；折返/小回环是否持续或突然出现；细节沿 s 是否平滑变化；同一来源是否仍混入重复段。绿色含义只有一种：该 FaceTrack 的原始几何。图中没有最终主轨、连接线、横向散点或补点。',
      '同组五栏共享 u/z 范围，绝对坐标不平移、不拉伸配准。局部窗口约 6.2 m 高；C/D/E/F/I/J 另附细节放大，仍保留五条邻线。图框边缘的截断只表示超出显示窗口，不代表原始分支终点。不同来源有各自编号及独立页面。编号仅在自己的局部区域有效。']
    for path in sorted((SOURCE/'region_data').glob('*.pkl')):
        data=pickle.load(path.open('rb'));r=data['region'];rid=r['region_id']
        by=defaultdict(list)
        for e in data['members']:
            if len(e['face_track_ids'])!=1:continue
            by[e['face_track_ids'][0]].append(e)
        blocks.append(f"## 案例 {rid}｜{descriptions[rid]}")
        for tid,members in sorted(by.items()):
            bounds=[r['bounds'][2],r['bounds'][3],r['bounds'][4],r['bounds'][5]]
            name=panel(out,r,tid,members,bounds)
            table=[]
            for key in r['target_keys']:
                em=[e for e in members if e['s']==key];bids=sorted({e['branch_id'] for e in em})
                faces={f for e in em for f in e['source_face_ids']}
                kinds=sorted({e['kind'] for e in em})
                row=dict(region_id=rid,face_track_id=tid,s=key,branch_ids=bids,edge_count=len(em),source_face_count=len(faces),
                     MBG_branch_ids=sorted({e['branch_id'] for e in em if e['MBG']}),branch_kinds=kinds,
                     filtered_by_current_route=False,manual_review='PENDING')
                records.append(row);table.append(f"| {key} | {', '.join('B'+str(b) for b in bids) or '无'} | {len(em)} | {len(faces)} |")
            blocks.extend([f'### {tid}',f'![{tid} 五条同来源分支](figures/{name})',
              '| s（m） | 原始分支编号 | 局部边数 | 来源 FaceID 数 |\n| --- | --- | --- | --- |\n'+'\n'.join(table)])
            group=dict(region_id=rid,face_track_id=tid,slices_with_members=len({e['s'] for e in members}),figure=name,
                       branches_by_slice={k:sorted({e['branch_id'] for e in members if e['s']==k}) for k in r['target_keys']},review='PENDING')
            if rid in zoom:
                low,high=zoom[rid];segments=[crop(e['points_uz'],[bounds[0],bounds[1],low,high]) for e in members]
                segments=[s for s in segments if s is not None]
                if segments:
                    points=np.concatenate(segments);umin,umax=points[:,0].min(),points[:,0].max();pad=max(.06,(umax-umin)*.07)
                    detail=panel(out,r,tid,members,[umin-pad,umax+pad,low,high],True)
                    group['detail_figure']=detail;blocks.append(f'![{tid} 同来源细节放大](figures/{detail})')
            if 'detail_figure' not in group:
                points=np.asarray([e['points_uz'] for e in members]).reshape(-1,2)
                lo,hi=points.min(axis=0),points.max(axis=0)
                if hi[1]-lo[1]<2.5:
                    pad=np.maximum((hi-lo)*.12,[.08,.10])
                    detail=panel(out,r,tid,members,[lo[0]-pad[0],hi[0]+pad[0],lo[1]-pad[1],hi[1]+pad[1]],True)
                    group['detail_figure']=detail
                    blocks.extend(['该来源只占原窗口的一小部分，下面单独放大；五条邻线仍使用同一坐标范围。',f'![{tid} 来源片段放大](figures/{detail})'])
            blocks.append('人工审核记录：□ 细节对应较好　□ 同来源仍有明显差异　□ 来源混杂/多段　□ 信息不足。具体位置：________。')
            groups.append(group);print('BRANCH_GROUP',tid,flush=True)
    csv_rows(out/'facetrack_branch_membership.csv',records)
    (out/'branch_gallery_index.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2),encoding='utf-8')
    blocks.extend(['## 如何解释审核结果',
      '若某组五条同来源分支的折返、平台和悬空相关细节能够对应，说明这个局部 FaceTrack 匹配提供了可用于下一步选轨的一致候选组。若同来源仍有明显差异，就应保留这个反例，继续检查真实曲面变化、重建伪影或 FaceTrack 合并过粗，不能直接把整组写进主轨。',
      '本轮完成的是按来源分组的证据展示；是否满足你需要的重建一致性，等待人工审核。不能仅因来源编号相同就宣布验证通过。'])
    (out/'FACETRACK_BRANCH_GALLERY.md').write_text('\n\n'.join(blocks)+'\n',encoding='utf-8')
    assert all(not r['filtered_by_current_route'] for r in records)
    print('BRANCH_GALLERY_COMPLETE',len(groups),len(records),flush=True)


if __name__=='__main__':main(latest())
