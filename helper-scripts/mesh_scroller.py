#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   mesh_scroller.py
@Time    :   2023/01/12 13:22:02
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Create a Dash app to scroll through a list of SMPL meshes (made to handle frankmocap output)
"""

import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize Dash app
app = dash.Dash()

# Read in file paths from txt file
with open("/home/sid/mesh_files.txt", "r") as f:
    filepaths = f.read().splitlines()

# Initialize mesh index
mesh_index = 0

# Create layout for Dash app
app.layout = html.Div([
    dcc.Graph(id="mesh_render"),
    html.Button("Previous", id="prev_button"),
    html.Button("Next", id="next_button")
])

# Define function to load mesh from file
def load_mesh(filepath):
    with open(filepath, "rb") as f:
        mesh = pickle.load(f)
    # Render mesh using Plotly
    pcloud = mesh['pred_output_list'][0]['pred_vertices_img']
    x,y,z = pcloud[:, 0], pcloud[:, 1], pcloud[:, 2]
    return {
        "data": [{
            "type": "mesh3d",
            "x": x,
            "y": y,
            "z": z,
            # "i": mesh["faces"][0],
            # "j": mesh["faces"][1],
            # "k": mesh["faces"][2],
            "colorbar": {
                "title": "Colorbar Title Goes Here",
                "thickness": 10
            }
        }]
    }

# Define callback for updating mesh render
@app.callback(
    Output("mesh_render", "figure"),
    [Input("prev_button", "n_clicks"), Input("next_button", "n_clicks")]
)
def update_mesh(prev_clicks, next_clicks):
    global mesh_index
    if prev_clicks is not None and prev_clicks > 0:
        mesh_index -= 1
        if mesh_index < 0:
            mesh_index = len(filepaths) - 1
    elif next_clicks is not None and next_clicks > 0:
        mesh_index += 1
        if mesh_index >= len(filepaths):
            mesh_index = 0

    return load_mesh(filepaths[mesh_index])

if __name__ == "__main__":
    app.run_server(debug=True)
