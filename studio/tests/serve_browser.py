"""Local-only browser QA server; test fixtures are never mounted in production."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import create_app, Engine
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
import os
import uvicorn
app = create_app(Engine())
app.router.routes.insert(0, Mount('/__test', app=StaticFiles(directory=Path(__file__).parent / 'browser', html=True)))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('FALCON_QA_PORT', 7866)))
