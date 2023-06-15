from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import os

deps_flow_router = APIRouter()

svg_html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVG Viewer</title>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        #container {{
            width: 100%;
            height: 100%;
            overflow: auto;
        }}
    </style>
</head>
<body>
    <div id="container">
        {svg_content}
    </div>
    <script>
        const container = document.getElementById('container');
        container.onmousedown = e => {{
            e.preventDefault();
            const startX = e.clientX;
            const startY = e.clientY;

            const onMove = e => {{
                container.scrollLeft = container.scrollLeft + startX - e.clientX;
                container.scrollTop = container.scrollTop + startY - e.clientY;
            }}

            const onUp = () => {{
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            }}

            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        }};
    </script>
</body>
</html>
"""


@deps_flow_router.get("/deps_flow", response_class=HTMLResponse)
async def get_deps_flow():
    html_content = "<h1>Modules</h1>"
    for filename in os.listdir("pydeps_data"):
        if filename.endswith(".svg"):
            module_name = filename[:-4]
            html_content += f'<a href="/deps_flow/{module_name}">{module_name}</a><br>'
    return HTMLResponse(content=html_content, status_code=200)


@deps_flow_router.get("/deps_flow/{module_name}", response_class=HTMLResponse)
async def get_svg(module_name: str):
    svg_path = os.path.join("pydeps_data", module_name + ".svg")
    if os.path.exists(svg_path) and os.path.isfile(svg_path):
        with open(svg_path, "r") as f:
            svg_content = f.read()
        html_content = svg_html_template.format(svg_content=svg_content)
        return HTMLResponse(content=html_content, status_code=200)

    else:
        return HTMLResponse(content="SVG not found", status_code=404)
