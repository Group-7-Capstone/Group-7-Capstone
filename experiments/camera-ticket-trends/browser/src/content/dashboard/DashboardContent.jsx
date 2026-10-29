import React, {useMemo,useState} from "react";
import {EvidenceChart,DataComponent,DataTable,Dropdown,Button,useDataApp,useDashboardTabs} from "../../data-app-public.jsx";
import {SharedTrends,CameraClusters} from "./SharedAnalysis.jsx";
import {RoadTrends} from "./RoadTrends.jsx";
import "./camera-trends.css";

const count=v=>v==null?"Unavailable":Number(v).toLocaleString("en-US");
const fixed=v=>v==null?"Not eligible":Number(v).toLocaleString("en-US",{maximumFractionDigits:1});
const date=v=>v?String(v).slice(0,10):"None observed";
const DAY=86400000;
function decode(payload,selected) {
  if(!payload||!selected) return [];
  const {start,values}=JSON.parse(payload);
  const first=Date.parse(start+"T00:00:00Z"), activation=Date.parse(date(selected.inferred_activation_date)+"T00:00:00Z");
  return values.map((v,i)=>({camera_id:selected.camera_id,week_start:new Date(first+i*7*DAY).toISOString().slice(0,10),
    weeks_since_activation:Math.floor((first+i*7*DAY-activation)/(7*DAY)),observed_ticket_count:v[0],ticket_count:v[1],
    moving_average_4w:v[2],is_complete_week:v[3],is_partial_week:v[4],source_complete:v[5],
    inferred_activation_date:date(selected.inferred_activation_date),last_actual_ticket_date:date(selected.last_actual_ticket_date)}));
}
function downloadCsv(rows,filename) {
  if(!rows.length) return;
  const fields=Object.keys(rows[0]), quote=v=>v==null?"":'"'+String(v).replaceAll('"','""')+'"';
  const csv=[fields,...rows.map(r=>fields.map(f=>r[f]))].map(r=>r.map(quote).join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
  const a=document.createElement("a"); a.href=url; a.download=filename; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function IndividualCameras() {
  const {reviewedRows,viewFocus}=useDataApp();
  const directory=reviewedRows("camera_directory",[]), payloads=reviewedRows("camera_series",[]), audit=reviewedRows("snapshot_status",[])[0]||{};
  const [type,setType]=useState("All types"),[confidence,setConfidence]=useState("All confidence labels"),[status,setStatus]=useState("All data statuses");
  const [selectedId,setSelectedId]=useState(viewFocus?.camera_id||null),[axis,setAxis]=useState("calendar");
  const filtered=useMemo(()=>directory.filter(r=>(type==="All types"||r.camera_type===type)&&(confidence==="All confidence labels"||r.detection_confidence===confidence)&&(status==="All data statuses"||r.data_status===status)),[directory,type,confidence,status]);
  const selected=filtered.find(r=>r.camera_id===selectedId)||filtered[0];
  const payload=payloads.find(r=>r.camera_id===selected?.camera_id)?.weekly_payload;
  const series=useMemo(()=>decode(payload,selected),[payload,selected]);
  const partial=series.filter(r=>r.is_partial_week);
  const columns=[{field:"search_query",label:"Address",presentation:"identity",secondaryField:"camera_id"},
    {field:"camera_type",label:"Type"},{field:"detection_confidence",label:"Original confidence"},
    {field:"inferred_activation_date",label:"Inferred activation",renderCell:date},
    {field:"last_actual_ticket_date",label:"Last actual ticket",renderCell:date},
    {field:"analysis_ticket_count",label:"Tickets",renderCell:count},{field:"data_status",label:"Data status"}];
  const spec={type:"line",x:axis==="calendar"?"week_start":"weeks_since_activation",y:"ticket_count",fields:["ticket_count","moving_average_4w"],
    stackable:false,yLabel:"Unique tickets per week",xLabel:axis==="calendar"?"Calendar week (Monday start)":"Weeks since inferred activation",valueDecimals:2,
    colors:{ticket_count:"var(--chart-1)",moving_average_4w:"var(--chart-2)"},legend:{labels:{ticket_count:"Weekly tickets",moving_average_4w:"4-week mean"}},
    annotations:axis==="calendar"&&series.length?[
      {id:"activation-week",kind:"event",at:series[0].week_start,field:"inferred_activation_date",label:`Inferred activation: ${date(selected.inferred_activation_date)}${series.length===1?`; last ticket: ${date(selected.last_actual_ticket_date)}`:""}`},
      ...(series.length>1?[{id:"last-ticket-week",kind:"event",at:series.at(-1).week_start,field:"last_actual_ticket_date",label:`Last ticket: ${date(selected.last_actual_ticket_date)}${series.at(-1).is_partial_week?"; partial week":""}`}]:[]),
      ...(!series[0].source_complete && series.some(r=>r.source_complete)?[{id:"early-coverage-gap",kind:"range",at:series[0].week_start,end:series.find(r=>r.source_complete).week_start,label:"Unavailable early coverage"}]:[])]:[]};
  return <article className="camera-trends">
    <header className="ct-intro"><p className="ct-eyebrow">NYC CAMERA ENFORCEMENT · OBSERVED TICKET HISTORIES</p>
      <h1>From inferred activation to the last observed ticket</h1>
      <p>Explore Eric’s 4,610 unchanged address/type groups. Issue dates: January 1, 2016–August 30, 2026. Counts describe recorded tickets; they do not measure violation rates or causal effects.</p></header>
    <DataComponent id="snapshot-coverage" queryId="snapshot_status" title="Snapshot coverage" kind="table" variant="plain" sourceRows={[audit]} displayRows={[audit]}>
      <div className="ct-coverage" data-reviewed-rows><span><strong>{count(directory.length)}</strong> groups listed</span><span><strong>{count(audit.analysis_ticket_count)}</strong> matched post-activation tickets</span><span className={audit.all_sources_complete?"ct-ready":"ct-caution"}>{audit.all_sources_complete?"All 11 requested files downloaded":"Historical download not yet complete"}</span></div>
      <p className="ct-footnote">{audit.coverage_note||"Source-file completeness and historical coverage are checked separately."}</p>
    </DataComponent>
    <DataComponent id="camera-directory" queryId="camera_directory" title="Camera directory" kind="table" variant="card" sourceRows={filtered} displayRows={filtered} description="Select a row to inspect its history below. Search by address or stable ID; sort using column headings.">
      <DataTable rows={filtered} columns={columns} label="Camera directory" searchable rowKey="camera_id" selectedRowKey={selected?.camera_id} onRowSelect={r=>setSelectedId(r.camera_id)} rowActionLabel={r=>`Inspect ${r.search_query}, ${r.camera_type}`} toolbarControls={<>
        <Dropdown label="Type" showLabel value={type} choices={["All types",...new Set(directory.map(r=>r.camera_type))]} onChange={setType}/>
        <Dropdown label="Confidence" showLabel value={confidence} choices={["All confidence labels",...new Set(directory.map(r=>r.detection_confidence))]} onChange={setConfidence}/>
        <Dropdown label="Data status" showLabel value={status} choices={["All data statuses",...new Set(directory.map(r=>r.data_status))]} onChange={setStatus}/>
      </>}/>
    </DataComponent>
    {selected?<section className="ct-detail" data-reviewed-rows>
      <div className="ct-selected-title"><p className="ct-eyebrow">SELECTED ENFORCEMENT GROUP · {selected.camera_id}</p><h2>{selected.search_query}</h2><p>{selected.camera_type} · {selected.detection_confidence} · {selected.data_status}</p></div>
      <DataComponent id="camera-facts" queryId="camera_directory" title="Observation window" kind="table" variant="plain" sourceRows={[selected]} displayRows={[selected]}><dl className="ct-facts">
        <div><dt>Tickets after activation</dt><dd>{count(selected.analysis_ticket_count)}</dd></div><div><dt>Inferred activation</dt><dd>{date(selected.inferred_activation_date)}</dd></div>
        <div><dt>Last actual ticket</dt><dd>{date(selected.last_actual_ticket_date)}</dd></div><div><dt>Observation span / zero-ticket weeks</dt><dd>{count(selected.observed_weeks)} / {count(selected.zero_ticket_weeks)}</dd></div>
      </dl></DataComponent>
      {series.length?<EvidenceChart key={axis} id={axis==="calendar"?"camera-calendar-trend":"camera-relative-trend"} queryId="camera_series" queryIds={["camera_series","camera_directory"]} title="Weekly ticket history" spec={spec} rows={series} sourceRows={series} height={380} variant="card" description={`${selected.search_query} | ${selected.camera_type} | ${selected.camera_id} | ${selected.detection_confidence}. ${count(selected.analysis_ticket_count)} observed tickets. Inferred activation: ${date(selected.inferred_activation_date)}; last actual ticket: ${date(selected.last_actual_ticket_date)}. Monday-start weeks; four-week mean requires complete weeks. Partial weeks: ${partial.map(r=>r.week_start).join(", ")||"none"}. Unavailable weeks: ${count(selected.unavailable_weeks)}. Counts are not rates or causal effects.`} headerControls={<>
        <Dropdown label="Time axis" value={axis} onChange={setAxis} choices={["calendar","relative"]} choiceLabels={{calendar:"Calendar",relative:"Relative weeks"}}/>
        <Button aria-label="Download weekly CSV" title="Download weekly CSV" onClick={()=>downloadCsv(series,`${selected.camera_id}_weekly.csv`)}>CSV</Button></>}>
        <p className="ct-footnote">Partial weeks: {partial.length?partial.map(r=>r.week_start).join(", "):"none"}. Unavailable weeks: {count(selected.unavailable_weeks)}. Both are excluded from smoothing and comparisons. Use the chart menu to export an image.</p>
      </EvidenceChart>:<div className="ct-empty">{selected.data_status}. A ticket history cannot be plotted for this group from the available snapshot.</div>}
      <div className="ct-lower">
        <DataComponent id="window-comparison" queryId="camera_directory" title="First versus last 13 complete weeks" kind="table" variant="card" sourceRows={[selected]} displayRows={[selected]}>
          <dl className="ct-comparison"><div><dt>First 13-week mean</dt><dd>{fixed(selected.first_13w_mean)}</dd></div><div><dt>Last 13-week mean</dt><dd>{fixed(selected.last_13w_mean)}</dd></div><div><dt>Absolute change</dt><dd>{fixed(selected.absolute_change)}</dd></div><div><dt>Percentage change</dt><dd>{selected.percentage_change==null?"Not available":fixed(selected.percentage_change)+"%"}</dd></div></dl>
          <p className="ct-footnote">{selected.comparison_start?`${selected.comparison_start}–${selected.comparison_end}: first and last 13 weeks of the longest fully covered run.`:"Requires at least 26 consecutive complete weeks."} A zero baseline has no percentage change.</p>
        </DataComponent>
        <DataComponent id="group-quality" queryId="camera_directory" title="Quality and original snapshot" kind="table" variant="card" sourceRows={[selected]} displayRows={[selected]}>
          <dl className="ct-quality"><div><dt>Original last observed week</dt><dd>{date(selected.last_ticket_date)}</dd></div><div><dt>New all-period / original tickets</dt><dd>{count(selected.new_total_all_dates)} / {count(selected.total_tickets_issued)}</dd></div><div><dt>Tickets before activation</dt><dd>{count(selected.tickets_before_activation)}</dd></div><div><dt>Original coordinates</dt><dd>{selected.Latitude}, {selected.Longitude}</dd></div></dl>
          <p className="ct-footnote">{selected.quality_flags||"No additional automated flags."} Fresh extracts may contain historical additions or revisions.</p>
        </DataComponent>
      </div>
    </section>:<p className="ct-empty">No groups match these filters.</p>}
    <footer className="ct-method">An enforcement group is a normalized address plus camera type, not a verified physical device. Eric’s normalization merges recorded directions. Inferred activation is not confirmed installation; the last ticket is not confirmed deactivation. A zero week means no observed ticket in covered data. Original grouping, confidence labels, coordinates, and activation dates remain unchanged.</footer>
  </article>;
}

export function DashboardContent() {
  const {activeTabId}=useDashboardTabs([
    {id:"dashboard",label:"Individual cameras",filterIds:[],focusFields:["camera_id"]},
    {id:"shared-trends",label:"Shared trends",filterIds:[]},
    {id:"road-trends",label:"Road characteristics",filterIds:[]},
    {id:"camera-clusters",label:"Camera clusters",filterIds:[]}
  ]);
  if(activeTabId==="shared-trends") return <SharedTrends/>;
  if(activeTabId==="road-trends") return <RoadTrends/>;
  if(activeTabId==="camera-clusters") return <CameraClusters/>;
  return <IndividualCameras/>;
}
