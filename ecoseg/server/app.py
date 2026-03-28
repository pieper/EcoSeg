"""EcoSeg FastAPI server.

Provides:
- REST API for experiment control (start, status, results)
- DICOMweb subset (WADO-RS, QIDO-RS) for OHIF viewer
- Static file serving for OHIF
- Dashboard with ECharts scatter plot
- Mosaic contact sheet endpoint
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ecoseg.experiments.runner import ExperimentRunner, ExperimentConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="EcoSeg", description="Ecological Segmentation Server")

# Global state
_runner: Optional[ExperimentRunner] = None
_experiment_task_running = False


class ExperimentRequest(BaseModel):
    config_path: Optional[str] = None
    data_root: str
    output_dir: str = "output"
    device: str = "auto"


# --- Experiment endpoints ---

@app.post("/api/experiment/start")
async def start_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
):
    """Start an experiment run in the background."""
    global _runner, _experiment_task_running

    if _experiment_task_running:
        raise HTTPException(409, "An experiment is already running")

    if request.config_path:
        config = ExperimentConfig.from_json(Path(request.config_path))
    else:
        config = ExperimentConfig()

    config.data_root = request.data_root
    config.output_dir = request.output_dir
    config.device = request.device

    _runner = ExperimentRunner(config)
    _experiment_task_running = True

    background_tasks.add_task(_run_experiment)

    return {"status": "started", "config": config.name}


async def _run_experiment():
    global _experiment_task_running
    try:
        _runner.run_full_experiment()
    except Exception as e:
        logger.exception("Experiment failed")
    finally:
        _experiment_task_running = False


@app.get("/api/experiment/status")
async def experiment_status():
    """Get current experiment status."""
    if _runner is None:
        return {"status": "no_experiment"}

    return {
        "status": "running" if _experiment_task_running else "complete",
        "generations_complete": len(_runner.results),
        "latest": _runner.results[-1].summary() if _runner.results else None,
    }


@app.get("/api/experiment/results")
async def experiment_results():
    """Get all generation results."""
    if _runner is None:
        return {"generations": []}

    return {
        "generations": [r.summary() for r in _runner.results],
        "per_scan": {
            gen.generation: [
                {"study_id": s.study_id, "dice": s.dice, "assd": s.assd}
                for s in gen.scores
            ]
            for gen in _runner.results
        },
    }


@app.get("/api/experiment/scores/{generation}")
async def generation_scores(generation: int):
    """Get per-scan scores for a specific generation."""
    if _runner is None or generation >= len(_runner.results):
        raise HTTPException(404, "Generation not found")

    result = _runner.results[generation]
    return {
        "generation": generation,
        "summary": result.summary(),
        "scores": [
            {"study_id": s.study_id, "dice": s.dice, "assd": s.assd}
            for s in result.scores
        ],
    }


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Interactive dashboard with ECharts scatter plot."""
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>EcoSeg Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        #scatter { width: 100%; height: 500px; }
        #convergence { width: 100%; height: 300px; margin-top: 20px; }
        .status { padding: 10px; background: #e8f5e9; border-radius: 4px; margin: 10px 0; }
        .controls { margin: 10px 0; }
        select { padding: 5px; font-size: 14px; }
    </style>
</head>
<body>
    <h1>EcoSeg Dashboard</h1>
    <div class="status" id="status">Loading...</div>
    <div class="controls">
        <label>Generation: <select id="genSelect" onchange="loadGeneration()"></select></label>
    </div>
    <div id="scatter"></div>
    <div id="convergence"></div>
    <script>
        const scatterChart = echarts.init(document.getElementById('scatter'));
        const convChart = echarts.init(document.getElementById('convergence'));

        async function refresh() {
            const status = await (await fetch('/api/experiment/status')).json();
            document.getElementById('status').textContent =
                `Status: ${status.status} | Generations: ${status.generations_complete}`;

            const results = await (await fetch('/api/experiment/results')).json();
            const select = document.getElementById('genSelect');
            select.innerHTML = '';
            results.generations.forEach(g => {
                const opt = document.createElement('option');
                opt.value = g.generation;
                opt.text = `Gen ${g.generation} (n=${g.num_training_scans})`;
                select.appendChild(opt);
            });
            if (results.generations.length > 0) {
                select.value = results.generations[results.generations.length - 1].generation;
                loadGeneration();
                updateConvergence(results.generations);
            }
        }

        async function loadGeneration() {
            const gen = document.getElementById('genSelect').value;
            const data = await (await fetch(`/api/experiment/scores/${gen}`)).json();

            const scatterData = data.scores.map(s => ({
                value: [s.dice, s.assd],
                name: s.study_id,
            }));

            scatterChart.setOption({
                title: { text: `Generation ${gen}: Dice vs ASSD` },
                tooltip: {
                    formatter: p => `${p.data.name}<br>Dice: ${p.data.value[0].toFixed(3)}<br>ASSD: ${p.data.value[1].toFixed(2)}mm`
                },
                xAxis: { name: 'Dice', min: 0, max: 1 },
                yAxis: { name: 'ASSD (mm)', inverse: true },
                series: [{
                    type: 'scatter',
                    data: scatterData,
                    symbolSize: 8,
                    itemStyle: { color: '#2196F3' },
                }]
            });

            scatterChart.off('click');
            scatterChart.on('click', params => {
                if (params.data && params.data.name) {
                    // Open OHIF viewer for this study
                    window.open(`/ohif/viewer?StudyInstanceUID=${params.data.name}`, '_blank');
                }
            });
        }

        function updateConvergence(generations) {
            convChart.setOption({
                title: { text: 'Convergence' },
                tooltip: { trigger: 'axis' },
                legend: { data: ['Mean Dice', 'Mean ASSD'] },
                xAxis: { type: 'category', data: generations.map(g => `Gen ${g.generation}`) },
                yAxis: [
                    { type: 'value', name: 'Dice', min: 0, max: 1 },
                    { type: 'value', name: 'ASSD (mm)', inverse: true },
                ],
                series: [
                    {
                        name: 'Mean Dice', type: 'line',
                        data: generations.map(g => g.mean_dice),
                        yAxisIndex: 0,
                    },
                    {
                        name: 'Mean ASSD', type: 'line',
                        data: generations.map(g => g.mean_assd),
                        yAxisIndex: 1,
                    },
                ]
            });
        }

        refresh();
        setInterval(refresh, 10000);
    </script>
</body>
</html>"""


# --- Entry point ---

def create_app(
    data_root: Optional[str] = None,
    ohif_dir: Optional[str] = None,
) -> FastAPI:
    """Create the FastAPI app with optional static file mounts."""
    if ohif_dir and Path(ohif_dir).exists():
        app.mount("/ohif", StaticFiles(directory=ohif_dir, html=True), name="ohif")

    return app
