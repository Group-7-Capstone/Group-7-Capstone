import React, {useState} from "react";
import {EvidenceChart,DataComponent,DataTable,Dropdown,Button,useDataApp} from "../../data-app-public.jsx";
import "./shared-analysis.css";
import "./road-trends.css";

const UNKNOWN="Unknown / ambiguous";
const names={lanes:"Travel lanes",width:"Paved road width",direction:"Traffic direction"};
const number=v=>v==null?"Unavailable":Number(v).toLocaleString("en-US",{maximumFractionDigits:1});
const colors=["#2864a5","#c46a25","#60804c","#98658d","#777d85"];
function saveCsv(rows,name){
  if(!rows.length)return;
  const fields=Object.keys(rows[0]),quote=v=>v==null?"":`"${String(v).replaceAll('"','""')}"`;
  const text=[fields,...rows.map(r=>fields.map(f=>r[f]))].map(r=>r.map(quote).join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob([text],{type:"text/csv;charset=utf-8"}));
  const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export function RoadTrends(){
  const {reviewedRows,exploreDashboard}=useDataApp();
  const [type,setType]=useState("Speed"),[horizon,setHorizon]=useState("52"),[metric,setMetric]=useState("index"),[smooth,setSmooth]=useState("4w");
  const [policy,setPolicy]=useState("intersection_only"),[baseline,setBaseline]=useState("0"),[showUnknown,setShowUnknown]=useState(false);
  const [dimension,setDimension]=useState("lanes"),[selectedGroup,setSelectedGroup]=useState("2 lanes"),[search,setSearch]=useState(""),[status,setStatus]=useState("All statuses");
  const accepts=r=>r.camera_type===type&&r.horizon===Number(horizon)&&r.policy===policy&&r.baseline_min===Number(baseline);
  const trends=reviewedRows("road_trends",[]).filter(accepts),coverage=reviewedRows("road_coverage",[]).filter(accepts);
  const allFeatures=reviewedRows("road_camera_features",[]),features=allFeatures.filter(r=>r.camera_type===type);
  const cohort=coverage.filter(r=>r.dimension==="lanes").reduce((n,r)=>n+r.cohort_n,0);
  const matched=features.filter(r=>r[`eligible_${horizon}w`]&&(policy==="intersection_only"?r.strict_match:r.any_match)).length;
  const measure=metric==="raw"?"Tickets / group / week":"Own first 13-week mean = 100";
  const value=`${metric}_mean${smooth==="4w"?"_4w":""}`;
  const detailGroups=coverage.filter(r=>r.dimension===dimension&&(metric==="raw"?r.cohort_n:r.index_n)>0).map(r=>r.group);
  const group=detailGroups.includes(selectedGroup)?selectedGroup:detailGroups[0];
  const detail=trends.filter(r=>r.dimension===dimension&&r.group===group);
  const directory=features.filter(r=>(status==="All statuses"||r.match_status===status)&&`${r.search_query} ${r.camera_id} ${r.matched_street_names}`.toLowerCase().includes(search.toLowerCase().trim()));
  const download=(rows,name)=><Button onClick={()=>saveCsv(rows,name)}>Download CSV</Button>;
  return <article className="camera-trends shared-analysis road-trends">
    <DataComponent id="road-trend-controls" queryId="road_coverage" title="Shared trends by road characteristics" kind="table" variant="plain" sourceRows={coverage} displayRows={coverage}
      headerControls={<>
        <Dropdown label="Camera type" showLabel value={type} onChange={v=>{setType(v);setStatus("All statuses");}} choices={["Speed","Red Light"]}/>
        <Dropdown label="Complete follow-up" showLabel value={horizon} onChange={setHorizon} choices={["26","52","104"]} choiceLabels={{26:"26 weeks",52:"52 weeks",104:"104 weeks"}}/>
        <Dropdown label="Measure" showLabel value={metric} onChange={setMetric} choices={["index","raw"]} choiceLabels={{index:"Normalized index",raw:"Weekly ticket count"}}/>
        <Dropdown label="Smoothing" showLabel value={smooth} onChange={setSmooth} choices={["4w","weekly"]} choiceLabels={{"4w":"Trailing four weeks",weekly:"Weekly mean"}}/>
      </>}>
      <p>Compare the same complete histories within each road category. Every address/type group receives equal weight. The normalized index is 100 × weekly tickets ÷ that group’s first 13-week mean.</p>
      <p className="ct-footnote">Road attributes describe the first named street near the original coordinate. The enforced approach is unverified. On two-way roads, travel lanes include both directions; parking lanes are excluded.</p>
      <div className="road-options">
        <Dropdown label="Road matching" showLabel value={policy} onChange={setPolicy} choices={["intersection_only","include_street_only"]} choiceLabels={{intersection_only:"Intersection street matches",include_street_only:"Also include street-only matches"}}/>
        <Dropdown label="Baseline sensitivity" showLabel value={baseline} onChange={setBaseline} choices={["0","5"]} choiceLabels={{0:"All baselines",5:"At least 5 tickets/week"}}/>
        <label><input type="checkbox" checked={showUnknown} onChange={e=>setShowUnknown(e.target.checked)}/> Show unknown / ambiguous curves</label>
      </div>
      <div className="sa-stats" data-reviewed-rows>
        <div><small>Original {type.toLowerCase()} groups</small><strong>{number(features.length)}</strong></div>
        <div><small>Complete histories after baseline filter</small><strong>{number(cohort)}</strong></div>
        <div><small>Road matches before baseline filter</small><strong>{number(matched)}</strong></div>
      </div>
      <p className="ct-footnote">Fixed cohorts select longer ticket histories. The supplied LION snapshot is not a reconstruction of road conditions at activation. Different road categories can differ in geography, activation year and traffic; these curves do not estimate road or camera effects.</p>
    </DataComponent>
    {Object.entries(names).map(([key,title])=>{
      const summary=coverage.filter(r=>r.dimension===key),unknown=summary.find(r=>r.group===UNKNOWN);
      const data=trends.filter(r=>r.dimension===key&&(showUnknown||r.group!==UNKNOWN)).map(r=>({...r,series_label:`${r.group} (n=${metric==="raw"?r.n:r.index_n})`}));
      const plotted=data.filter(r=>(metric==="raw"?r.n:r.index_n)>0);
      const map=Object.fromEntries(plotted.map(r=>[r.series_label,colors[r.group===UNKNOWN?4:r.group_order]]));
      const small=summary.filter(r=>r.group!==UNKNOWN&&(metric==="raw"?r.cohort_n:r.index_n)>0&&(metric==="raw"?r.cohort_n:r.index_n)<10);
      return <EvidenceChart key={key} id={`road-trends-${key}`} queryId="road_trends" title={`${title} · ${type}`} rows={plotted} sourceRows={plotted} height={325} variant="card"
        description={`${horizon} complete relative weeks; equal-group means. ${smooth==="4w"?"Trailing four-week average; first three weeks unavailable.":"Unsmoothed weekly means."} Pointwise bootstrap bounds are available in the detail chart below.`}
        spec={{type:"line",x:"week",y:value,series:"series_label",colors:map,stackable:false,valueDecimals:2,xLabel:"Weeks from inferred activation (first full week = 1)",yLabel:measure}}>
        {!plotted.length&&<p className="ct-empty">No resolved groups with complete histories for this selection.</p>}
        <p className="ct-footnote">{number(cohort-(unknown?.cohort_n||0))} of {number(cohort)} complete histories have a resolved {title.toLowerCase()} category; {number(unknown?.cohort_n)} are unknown or ambiguous. {key==="width"?"Both minimum and maximum segment widths must fall in the same fixed band; intervals crossing a boundary remain unknown. ":""}{small.length?`Small groups (fewer than 10): ${small.map(r=>`${r.group}, n=${metric==="raw"?r.cohort_n:r.index_n}`).join("; ")}. `:""}{metric==="index"?"Zero baselines are excluded from the index; very small positive baselines can amplify changes.":"Counts are not traffic-adjusted rates."}</p>
        <div className="road-chart-download">{download(plotted,`${type.replaceAll(" ","_")}_${key}_${horizon}w_${metric}.csv`)}</div>
      </EvidenceChart>;
    })}
    <div className="road-options"><Dropdown label="Road characteristic" showLabel value={dimension} onChange={setDimension} choices={Object.keys(names)} choiceLabels={names}/><Dropdown label="Road category" showLabel value={group||"Unavailable"} onChange={setSelectedGroup} choices={detailGroups.length?detailGroups:["Unavailable"]}/></div>
    <EvidenceChart id="road-trends-uncertainty" queryId="road_trends" title="Weekly variation and uncertainty within a road category" rows={detail} sourceRows={detail} height={310} variant="card"
      spec={{type:"line",x:"week",y:`${metric}_mean`,fields:[`${metric}_mean`,`${metric}_median`,`${metric}_ci_low`,`${metric}_ci_high`],stackable:false,valueDecimals:2,xLabel:"Relative week",yLabel:measure,
        colors:{[`${metric}_mean`]:colors[0],[`${metric}_median`]:colors[1],[`${metric}_ci_low`]:"#777d85",[`${metric}_ci_high`]:"#777d85"},
        legend:{labels:{[`${metric}_mean`]:"Weekly mean",[`${metric}_median`]:"Weekly median",[`${metric}_ci_low`]:"95% CI lower",[`${metric}_ci_high`]:"95% CI upper"}}}}>
      <p className="ct-footnote">{group||"No group"}; n={number(detail[0]?.[metric==="raw"?"n":"index_n"])}. This detail chart always shows unsmoothed weeks. Bounds are pointwise 95% intervals from 500 resamples of whole histories. They do not capture shared-intersection dependence, mapping uncertainty or historical road changes; unavailable for n&lt;2. Median lines reveal sensitivity to high-volume or high-index groups.</p>
      <div className="road-chart-download">{download(detail,"road_category_uncertainty.csv")}</div>
    </EvidenceChart>
    <DataComponent id="road-cohort-counts" queryId="road_coverage" title="Category sizes and first-to-last-window comparison" kind="table" variant="card" sourceRows={coverage} displayRows={coverage} headerControls={download(coverage,"road_category_coverage.csv")}>
      <DataTable rows={coverage} rowKey={r=>`${r.dimension}-${r.group}`} label="Road category coverage" searchable={false} columns={[
        {field:"dimension",label:"Characteristic",renderCell:v=>names[v]},{field:"group",label:"Category"},{field:"cohort_n",label:"Raw n"},{field:"index_n",label:"Index n"},
        {field:"first_13w_raw_mean",label:"First 13 weeks: tickets/week",renderCell:number},{field:"last_13w_raw_mean",label:"Last 13 weeks: tickets/week",renderCell:number},
        {field:"last_13w_index_mean",label:"Last 13 weeks: mean index",renderCell:number},{field:"low_baseline_n",label:"Baseline below 5"}]}/>
      <p className="ct-footnote">Within each characteristic, known plus unknown groups reconcile to the full fixed cohort. Characteristics reuse the same groups and must not be added together. Empty categories have n=0 and unavailable estimates.</p>
    </DataComponent>
    <DataComponent id="road-match-directory" queryId="road_camera_features" title="Road matching directory · all original groups" kind="table" variant="card" sourceRows={directory} displayRows={directory} headerControls={download(directory,"camera_road_feature_matches.csv")}>
      <DataTable rows={directory} rowKey="camera_id" label="Road feature match directory" searchable={false} onRowSelect={r=>exploreDashboard("dashboard",{focus:{camera_id:r.camera_id}})} rowActionLabel={r=>`Open ticket history for ${r.search_query}`}
        toolbarControls={<><input className="sa-search" aria-label="Search road matches" placeholder="Search address or camera ID" value={search} onChange={e=>setSearch(e.target.value)}/><Dropdown label="Match status" showLabel value={status} onChange={setStatus} choices={["All statuses",...new Set(features.map(r=>r.match_status))]}/></>}
        columns={[{field:"search_query",label:"Address",presentation:"identity",secondaryField:"camera_id"},{field:"match_status",label:"Match status"},{field:"lanes_group",label:"Travel lanes"},{field:"width_group",label:"Width"},{field:"direction_group",label:"Direction"},{field:"match_distance_m",label:"Distance, m",renderCell:number},{field:"candidate_segments",label:"Segments"}]}/>
      <p className="ct-footnote">The directory shows the original matching result, independent of the follow-up, baseline and matching-policy controls. Street-only results enter curves only when explicitly included. Multiple candidate segments contribute one camera history, not several. Matching confidence differs from Eric’s activation confidence.</p>
    </DataComponent>
    <details className="road-methods"><summary>Matching rules and interpretation</summary>
      <p>Names are normalized only for this join: abbreviations, numeric ordinals and split words. Camera IDs, original names, coordinates, activation dates, ticket counts and existing clusters are unchanged. No fuzzy name guesses or arbitrary nearest-road assignments are used.</p>
      <p>Intersection matches require both named streets within 60 m of the original coordinate, a compatible ticket borough, and connected LION node IDs with matching level codes. Only ordinary B roadbeds are assigned. Divided roads, generic/roadbed ambiguity and nonstandard road types remain unresolved because enforced direction is unknown.</p>
      <p>When adjacent segments differ, an attribute is assigned only if every candidate agrees on its category. Width is paved roadway width, not lane width, right-of-way width or pedestrian crossing distance. Source release vintage has not been independently established.</p>
      <p>Ticket snapshot: October 14, 2026; issue-date cutoff: August 30, 2026. Complete-week coverage begins July 4, 2016. Results describe observed ticket counts and may reflect traffic, enforcement, calendar and selection differences.</p>
    </details>
  </article>;
}
