import React,{useState} from "react";
import {EvidenceChart,DataComponent,DataTable,Dropdown,Button,useDataApp} from "../../data-app-public.jsx";
import "./shared-analysis.css";

const number=v=>v==null?"Unavailable":Number(v).toLocaleString("en-US",{maximumFractionDigits:1});
const percent=v=>v==null?"Unavailable":`${number(v)}%`;
const types=["Speed","Red Light"];
const palette=["var(--chart-1)","var(--chart-2)","var(--chart-3)","var(--chart-4)","var(--chart-5)"];
function csv(rows,name) {
  if(!rows.length)return;
  const fields=Object.keys(rows[0]),quote=v=>v==null?"":`"${String(v).replaceAll('"','""')}"`;
  const text=[fields,...rows.map(r=>fields.map(f=>r[f]))].map(r=>r.map(quote).join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob([text],{type:"text/csv;charset=utf-8"}));
  const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const line=(x,y,fields,labels,yLabel)=>({type:"line",x,y,fields,stackable:false,valueDecimals:2,xLabel:x==="week"?"Weeks from inferred activation (first full week = 1)":"Calendar week",yLabel,legend:{labels},colors:Object.fromEntries(fields.map((f,i)=>[f,palette[i%palette.length]]))});
function Export({rows,name}) {return <Button onClick={()=>csv(rows,name)} aria-label={`Download ${name}`}>CSV</Button>;}
function TypeChoice({value,onChange}) {return <Dropdown label="Camera type" showLabel value={value} onChange={onChange} choices={types}/>;}

export function SharedTrends() {
  const {reviewedRows}=useDataApp();
  const [type,setType]=useState("Speed"),[scenario,setScenario]=useState("balanced_52"),[metric,setMetric]=useState("index");
  const all=reviewedRows("analysis_shared_trends",[]),calendar=reviewedRows("analysis_calendar_trends",[]).filter(r=>r.camera_type===type);
  const coverage=reviewedRows("analysis_coverage",[]).filter(r=>r.camera_type===type);
  const rows=all.filter(r=>r.camera_type===type&&r.scenario===scenario);
  const choices=[...new Map(all.filter(r=>!r.scenario.startsWith("cohort_")).map(r=>[r.scenario,r.scenario_label])).entries()];
  const first=rows[0],unit=metric==="raw"?"Tickets / camera / week":"Own 13-week baseline = 100";
  const meanFields=[`${metric}_mean`,`${metric}_mean_4w`,`${metric}_ci_low`,`${metric}_ci_high`];
  const labels={[`${metric}_mean`]:"Mean",[`${metric}_mean_4w`]:"Four-week mean",[`${metric}_ci_low`]:"95% CI lower",[`${metric}_ci_high`]:"95% CI upper"};
  const comparison=all.filter(r=>r.camera_type===type&&["balanced_52","high_confidence","administrative","skip_first_four","baseline_at_least_five"].includes(r.scenario));
  const comparisonColors=Object.fromEntries(["balanced_52","high_confidence","administrative","skip_first_four","baseline_at_least_five"].map((s,i)=>[all.find(r=>r.scenario===s)?.scenario_label,["#2864a5","#c46a25","#60804c","#98658d","#958128"][i]]));
  const cohorts=all.filter(r=>r.camera_type===type&&r.scenario.startsWith("cohort_"));
  return <article className="camera-trends shared-analysis">
    <DataComponent id="shared-controls" queryId="analysis_shared_trends" queryIds={["analysis_shared_trends","analysis_coverage"]} title="Shared ticket trajectories" kind="table" variant="plain" sourceRows={rows} displayRows={rows} headerControls={<>
      <TypeChoice value={type} onChange={setType}/><Dropdown label="Analysis cohort" showLabel value={scenario} onChange={setScenario} choices={choices.map(x=>x[0])} choiceLabels={Object.fromEntries(choices)}/>
      <Dropdown label="Measure" showLabel value={metric} onChange={setMetric} choices={["index","raw"]} choiceLabels={{index:"Normalized index",raw:"Weekly ticket count"}}/>
    </>}>
      <p>Each enforcement group receives equal weight. {scenario==="available"?"Contributors vary by week; follow the denominator chart.":"The same eligible groups contribute throughout the selected window."} Speed and red-light locations are analyzed separately.</p>
      <div className="sa-stats" data-reviewed-rows><div><small>Raw contributors at first week</small><strong>{number(first?.n)}</strong></div><div><small>Indexed contributors at first week</small><strong>{number(first?.index_n)}</strong></div><div><small>Follow-up shown</small><strong>{first?`Weeks ${first.week}–${rows.at(-1).week}`:"Unavailable"}</strong></div></div>
      <p className="ct-footnote">{number(first?.cohort_groups)} of {number(coverage.reduce((sum,r)=>sum+r.groups,0))} original {type.toLowerCase()} groups enter this view. The index is 100 × weekly tickets ÷ that camera’s first 13-week mean. Zero or incomplete baselines are excluded only from the index. {scenario==="skip_first_four"?"This sensitivity uses weeks 5–17 as its baseline.":""} {scenario==="administrative"?"Post-last-ticket zeros mean no observed tickets through the fixed cutoff; continued operation is unverified.":"Complete weeks stop at the last observed ticket; eligibility selects longer histories."}</p>
    </DataComponent>
    {!rows.length?<p className="ct-empty">No complete histories for this selection.</p>:<>
      <EvidenceChart id={`shared-mean-${metric}`} queryId="analysis_shared_trends" title={`${type} mean · ${first.scenario_label}`} rows={rows} sourceRows={rows} height={350} variant="card" description={`Raw contributors at the first shown week: ${first.n}; indexed contributors: ${first.index_n}. Issue-date cutoff August 30, 2026. Pointwise 95% bootstrap intervals resample whole histories. Counts are descriptive, not causal effects.`} spec={line("week",`${metric}_mean`,meanFields,labels,unit)} headerControls={<Export rows={rows} name={`${type.toLowerCase().replaceAll(" ","_")}_${scenario}_${metric}.csv`}/>}>
        <p className="ct-footnote">Lower and upper lines bound the pointwise 95% bootstrap interval for the weekly mean. The 500 resamples draw whole camera histories; spatial and shared-calendar dependence is not modeled. The smoothed line is a trailing four-week mean.</p>
      </EvidenceChart>
      <div className="sa-grid">
        <EvidenceChart id={`shared-spread-${metric}`} queryId="analysis_shared_trends" title="Across-camera variation" rows={rows} sourceRows={rows} height={270} variant="card" spec={line("week",`${metric}_median`,[`${metric}_q25`,`${metric}_median`,`${metric}_q75`],{[`${metric}_q25`]:"25th percentile",[`${metric}_median`]:"Median",[`${metric}_q75`]:"75th percentile"},unit)}><p className="ct-footnote">The middle 50% of camera observations describes heterogeneity, not uncertainty in the mean.</p></EvidenceChart>
        <EvidenceChart id="shared-contributors" queryId="analysis_shared_trends" title="Contributing enforcement groups" rows={rows} sourceRows={rows} height={270} variant="card" spec={line("week","n",["n","index_n"],{n:"Raw counts",index_n:"Normalized index"},"Groups")}/>
      </div>
    </>}
    <EvidenceChart id="shared-sensitivities" queryId="analysis_shared_trends" title="52-week sensitivity comparisons" rows={comparison} sourceRows={comparison} height={340} variant="card" spec={{type:"line",x:"week",y:"index_mean",series:"scenario_label",colors:comparisonColors,stackable:false,valueDecimals:2,yLabel:"Mean index",xLabel:"Relative week"}} headerControls={<Export rows={comparison} name="shared_sensitivities.csv"/>}>
      <p className="ct-footnote">These comparisons use {type.toLowerCase()} groups and their stated cohorts. Omitting weeks 1–4 changes the baseline to weeks 5–17. A baseline below five tickets/week can amplify relative changes. Activation is activity-based, so an early decline can reflect selection.</p>
    </EvidenceChart>
    <EvidenceChart id="shared-activation-cohorts" queryId="analysis_shared_trends" title="52-week trends by inferred activation year" rows={cohorts} sourceRows={cohorts} height={330} variant="card" spec={{type:"line",x:"week",y:"index_mean_4w",series:"scenario_label",stackable:false,valueDecimals:2,yLabel:"Four-week mean index",xLabel:"Relative week"}}>
      <p className="ct-footnote">Each year has its own fixed cohort. Exact cohort sizes accompany the exported data; very small cohorts can be unstable.</p>
    </EvidenceChart>
    <div className="sa-grid">
      <EvidenceChart id="shared-calendar-counts" queryId="analysis_calendar_trends" title="Calendar-time ticket counts" rows={calendar} sourceRows={calendar} height={285} variant="card" spec={line("week_start","raw_mean",["raw_mean","raw_median"],{raw_mean:"Mean",raw_median:"Median"},"Tickets / camera / week")} headerControls={<Export rows={calendar} name="calendar_shared_trend.csv"/>}/>
      <EvidenceChart id="shared-calendar-contributors" queryId="analysis_calendar_trends" title="Calendar-time contributors" rows={calendar} sourceRows={calendar} height={285} variant="card" spec={line("week_start","n",["n"],{n:"Complete-week contributors"},"Groups")}/>
    </div>
    <DataComponent id="shared-eligibility" queryId="analysis_coverage" title="Eligibility by original confidence" kind="table" variant="card" sourceRows={coverage} displayRows={coverage}>
      <DataTable rows={coverage} rowKey="detection_confidence" label="Eligibility by confidence" columns={[{field:"detection_confidence",label:"Original confidence"},{field:"groups",label:"All groups"},{field:"eligible_26w",label:"26 weeks"},{field:"eligible_52w",label:"52 weeks"},{field:"eligible_104w",label:"104 weeks"},{field:"eligible_administrative_52w",label:"52 weeks: fixed cutoff"}]}/>
    </DataComponent>
    <p className="ct-method">Frozen October 14, 2026 extract; issue dates through August 30, 2026. Original groups and activation dates are unchanged. Weeks before July 4, 2016 lack assured source coverage. Counts describe recorded tickets and do not establish violation rates, driving improvement, or effects on collisions.</p>
  </article>;
}

export function CameraClusters() {
  const {reviewedRows,exploreDashboard}=useDataApp();
  const [type,setType]=useState("Red Light"),[cluster,setCluster]=useState("All clusters"),[metric,setMetric]=useState("shape_mean"),[status,setStatus]=useState("All statuses"),[search,setSearch]=useState("");
  const profiles=reviewedRows("analysis_cluster_profiles",[]).filter(r=>r.camera_type===type);
  const selectedCluster=profiles.some(r=>r.cluster_id===cluster)?cluster:"All clusters";
  const selectedProfiles=profiles.filter(r=>selectedCluster==="All clusters"||r.cluster_id===selectedCluster);
  const curves=reviewedRows("analysis_cluster_curves",[]).filter(r=>r.camera_type===type&&(selectedCluster==="All clusters"||r.cluster_id===selectedCluster));
  const model=reviewedRows("analysis_models",[]).find(r=>r.camera_type===type);
  const diagnostics=reviewedRows("analysis_cluster_diagnostics",[]).filter(r=>r.camera_type===type);
  const sensitivity=reviewedRows("analysis_cluster_sensitivity",[]).filter(r=>r.camera_type===type);
  const members=reviewedRows("analysis_cluster_membership",[]).filter(r=>r.camera_type===type&&(selectedCluster==="All clusters"||r.cluster_id===selectedCluster)&&(status==="All statuses"||r.cluster_status===status)&&`${r.camera_id} ${r.search_query}`.toLowerCase().includes(search.trim().toLowerCase()));
  const examples=reviewedRows("analysis_cluster_examples",[]).filter(r=>r.camera_type===type&&(selectedCluster==="All clusters"||r.cluster_id===selectedCluster));
  const composition=reviewedRows("analysis_cluster_composition",[]).filter(r=>r.camera_type===type&&(selectedCluster==="All clusters"||r.cluster_id===selectedCluster));
  const inspect=r=>exploreDashboard("dashboard",{focus:{camera_id:r.camera_id}});
  return <article className="camera-trends shared-analysis">
    <DataComponent id="cluster-controls" queryId="analysis_models" queryIds={["analysis_models","analysis_cluster_membership"]} title="Camera trajectory clusters" kind="table" variant="plain" sourceRows={model?[model]:[]} displayRows={model?[model]:[]} headerControls={<><TypeChoice value={type} onChange={v=>{setType(v);setCluster("All clusters");setStatus("All statuses");setSearch("");}}/><Dropdown label="Cluster" showLabel value={selectedCluster} onChange={setCluster} choices={["All clusters",...profiles.map(r=>r.cluster_id)]} choiceLabels={Object.fromEntries(profiles.map(r=>[r.cluster_id,r.cluster_label]))}/></>}>
      <p>Clusters compare the shape of the first 52 complete weeks. Four-week means are log-transformed and standardized within each group, so volume is reported separately.</p>
      <div className="sa-stats" data-reviewed-rows><div><small>Modeled groups</small><strong>{number(model?.modeled_groups)}</strong></div><div><small>Selected clusters</small><strong>{model?.selected_k==null?"None accepted":number(model.selected_k)}</strong></div><div><small>{model?.selected_k==null?"Best candidate separation / stability":"Separation / stability"}</small><strong>{Number(model?.silhouette??model?.best_candidate_silhouette).toFixed(3)} / {Number(model?.stability_ari??model?.best_candidate_stability).toFixed(3)}</strong></div></div>
      <p className="ct-footnote">{number(model?.modeled_groups)} of {number(reviewedRows("analysis_cluster_membership",[]).filter(r=>r.camera_type===type).length)} original {type.toLowerCase()} groups are modeled. These are exploratory shape groups, not verified operational categories. Sparse histories are retained; incomplete and constant block profiles remain in the directory with explicit statuses.</p>
    </DataComponent>
    {curves.length?<EvidenceChart id={`cluster-trajectories-${metric}`} queryId="analysis_cluster_curves" title="Cluster trajectories" rows={curves} sourceRows={curves} height={365} variant="card" spec={{type:"line",x:"week",y:metric,series:"cluster_label",stackable:false,valueDecimals:2,yLabel:metric==="shape_mean"?"Standardized log four-week mean":metric==="mean"?"Tickets / camera / week":"Own first 13-week mean = 100",xLabel:"Relative week"}} headerControls={<><Dropdown label="Trajectory measure" showLabel value={metric} onChange={setMetric} choices={["shape_mean","index_median","mean"]} choiceLabels={{shape_mean:"Standardized shape",index_median:"Median normalized index",mean:"Mean ticket count"}}/><Export rows={curves} name="cluster_trajectories.csv"/></>}/>:<p className="ct-empty">No candidate clustering solution passed all three selection criteria. The best candidate has K={model?.best_candidate_k}, silhouette {Number(model?.best_candidate_silhouette).toFixed(3)} and stability ARI {Number(model?.best_candidate_stability).toFixed(3)}. The diagnostics below distinguish weak separation from unstable assignment.</p>}
    {selectedProfiles.length>0&&<DataComponent id="cluster-profiles" queryId="analysis_cluster_profiles" title="Cluster profiles" kind="table" variant="card" sourceRows={selectedProfiles} displayRows={selectedProfiles} headerControls={<Export rows={selectedProfiles} name="cluster_profiles.csv"/>}>
      <DataTable rows={selectedProfiles} searchable={false} rowKey="cluster_id" label="Cluster profiles" columns={[{field:"cluster_label",label:"Shape"},{field:"groups",label:"Groups"},{field:"median_weekly_tickets",label:"Median weekly volume",renderCell:number},{field:"median_change_pct",label:"Median first/last 13-week change",renderCell:percent},{field:"high_confidence_groups",label:"High sustained confidence"}]}/>
    </DataComponent>}
    {examples.length>0&&<EvidenceChart id="cluster-representatives" queryId="analysis_cluster_examples" title="Representative individual trajectories" rows={examples} sourceRows={examples} height={300} variant="card" spec={{type:"line",x:"week",y:"index",series:"camera_id",stackable:false,valueDecimals:2,yLabel:"Own first 13-week mean = 100",xLabel:"Relative week"}}>
      <p className="ct-footnote">Three actual groups nearest each centroid. Search their identifiers in the directory below to open their full original histories.</p>
    </EvidenceChart>}
    <DataComponent id="cluster-member-directory" queryId="analysis_cluster_membership" title="Camera membership and eligibility" kind="table" variant="card" sourceRows={members} displayRows={members} headerControls={<Export rows={members} name="cluster_membership.csv"/>}>
      <DataTable rows={members} searchable={false} rowKey="camera_id" label="Cluster member directory" onRowSelect={inspect} rowActionLabel={r=>`Open full history for ${r.search_query}`} columns={[{field:"search_query",label:"Address",presentation:"identity",secondaryField:"camera_id"},{field:"cluster_label",label:"Cluster"},{field:"cluster_status",label:"Status"},{field:"exclusion_reason",label:"Eligibility"},{field:"detection_confidence",label:"Original confidence"},{field:"weekly_mean_52w",label:"Weekly volume",renderCell:number}]} toolbarControls={<><input className="sa-search" aria-label="Search cameras" placeholder="Search address or camera ID" value={search} onChange={e=>setSearch(e.target.value)}/><Dropdown label="Membership status" showLabel value={status} onChange={setStatus} choices={["All statuses",...new Set(reviewedRows("analysis_cluster_membership",[]).map(r=>r.cluster_status))]}/></>}/>
    </DataComponent>
    <div className="sa-grid">
      <DataComponent id="cluster-candidate-diagnostics" queryId="analysis_cluster_diagnostics" title="Model selection by camera type" kind="table" variant="card" sourceRows={diagnostics} displayRows={diagnostics}>
        <DataTable rows={diagnostics} rowKey="k" label="Clustering diagnostics" columns={[{field:"k",label:"K"},{field:"silhouette",label:"Silhouette",renderCell:v=>Number(v).toFixed(3)},{field:"stability_ari",label:"Stability ARI",renderCell:v=>Number(v).toFixed(3)},{field:"smallest_cluster",label:"Smallest group"},{field:"selected",label:"Selected"}]}/>
        <p className="ct-footnote">Select the highest silhouette with silhouette ≥0.20, mean stability ARI ≥0.75, and every cluster ≥max(10, 2% of modeled groups). Stability uses 12 fits on 80% subsamples. These diagnostics cover all {type.toLowerCase()} model members.</p>
      </DataComponent>
      <DataComponent id="cluster-robustness" queryId="analysis_cluster_sensitivity" title="Robustness checks" kind="table" variant="card" sourceRows={sensitivity} displayRows={sensitivity}>
        <DataTable rows={sensitivity} rowKey="sensitivity" label="Clustering robustness" columns={[{field:"sensitivity",label:"Refit"},{field:"groups",label:"Groups"},{field:"ari",label:"Agreement ARI",renderCell:v=>v==null?"Unavailable":Number(v).toFixed(3)}]}/>
        <p className="ct-footnote">ARI compares membership after refitting; 1 means identical partitions. Low agreement signals sensitivity. Original confidence labels refer to inferred activation, not clustering accuracy.</p>
      </DataComponent>
    </div>
    <DataComponent id="cluster-composition" queryId="analysis_cluster_composition" title="Cluster composition by activation year and confidence" kind="table" variant="card" sourceRows={composition} displayRows={composition} headerControls={<Export rows={composition} name="cluster_composition.csv"/>}>
      <DataTable rows={composition} searchable={false} label="Cluster composition" columns={[{field:"cluster_label",label:"Cluster"},{field:"activation_year",label:"Activation year"},{field:"detection_confidence",label:"Original confidence"},{field:"groups",label:"Groups"}]}/>
    </DataComponent>
    <p className="ct-method">No camera IDs, activation dates, coordinates or ticket counts were changed. A declining ticket trajectory does not establish improved driving or a causal safety benefit. Files and analysis remain local.</p>
  </article>;
}
