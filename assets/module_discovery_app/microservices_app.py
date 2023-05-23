from dash import Dash, html, Input, Output, dcc, ctx, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

# Import the module data as a dataframe
import module_data
df = module_data.df

# Import styling from assets directory
from assets import default_stylesheet 

# Import app components and their internal callbacks
from components.center_nav_bar import center_nav_bar, center_nav_bar_callbacks 
center_nav_bar = center_nav_bar.center_nav_bar

from components.visualization_panel import visualization_panel
visualization_panel = visualization_panel.visualization_panel

from components.app_title import app_title
app_title = app_title.app_title

from components.heading_tabs import heading_tabs
heading_tabs = heading_tabs.heading_tabs


# Import inter-component callbacks
import turn_nodes_on_off_callbacks

import filter_modules
filter_modules = filter_modules.filter_modules



# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Set up the layout of the app
app.layout = html.Div([
    dbc.Row(children=[
        app_title,
        heading_tabs]
        ),
    html.Hr(),
    dbc.Row(children=[
        visualization_panel,
        center_nav_bar,
        #information_panel
        ]),
    html.Div("none selected", id= "debugging_helper")

    ],
    style={'padding' : '25px'}
    )

@app.callback(Output('debugging_helper', 'children'),
              Input('general_options_checklist', 'value'),
            Input('coding_level_checklist', 'value'),
              )
def get_active_node(value, coding_level):
    return filter_modules(value)


# Initialize all INTRAcomponent callbacks
center_nav_bar_callbacks.get_center_nav_bar_callbacks(app)

# Initialize all INTERcomponent callbacks next...
turn_nodes_on_off_callbacks.turn_nodes_on_off(app)
    
if __name__ == '__main__':
    app.run_server(debug=True)