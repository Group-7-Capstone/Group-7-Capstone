"""Check that the browser and executed notebook contain the validated outputs."""
import json
import numpy as np
import pandas as pd
import nbformat
from preprocessing.camera_ticket_trends import DELIVERY, OUT


def validate():
    summary=pd.read_csv(OUT/"camera_summary.csv")
    weekly=pd.read_parquet(OUT/"camera_weekly_trends.parquet")
    snapshot=json.loads((DELIVERY/"browser/src/data.json").read_text(encoding="utf-8"))
    catalog=pd.DataFrame(snapshot["queries"]["camera_directory"]["rows"]).set_index("camera_id")
    assert set(catalog.index)==set(summary.camera_id)
    grouped=dict(tuple(weekly.groupby("camera_id",sort=False)))
    checked=0
    fields=["observed_ticket_count","ticket_count","moving_average_4w","is_complete_week","is_partial_week","source_complete"]
    for encoded in snapshot["queries"]["camera_series"]["rows"]:
        camera_id=encoded["camera_id"]
        payload=json.loads(encoded["weekly_payload"])
        source=grouped[camera_id]
        actual=pd.DataFrame(payload["values"],columns=fields)
        assert payload["start"]==str(source.week_start.iloc[0].date())
        assert np.allclose(actual.to_numpy(dtype=float),source[fields].to_numpy(dtype=float),equal_nan=True)
        assert int(actual.observed_ticket_count.sum())==catalog.loc[camera_id,"analysis_ticket_count"]
        checked+=len(actual)
    nb=nbformat.read(DELIVERY/"Per_Camera_Ticket_Trends.ipynb",as_version=4)
    code=[c for c in nb.cells if c.cell_type=="code"]
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    errors=[o for c in code for o in c.outputs if o.output_type=="error"]
    assert not errors
    images=sum("image/png" in o.get("data",{}) for c in code for o in c.outputs)
    assert images==7
    result={"browser_groups_checked":len(catalog),"browser_weekly_rows_checked":checked,
            "browser_values_match_parquet":True,"executed_code_cells":len(code),"notebook_error_outputs":len(errors),
            "embedded_figures":images,"exported_pngs":len(list((OUT/"figures").glob("*.png")))}
    (OUT/"artifact_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    validate()
