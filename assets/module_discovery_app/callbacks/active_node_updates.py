### Take the ACTIVE NODE and do everything we want to do with that information (displayinformation, update stylsheet, etc)
from dash import Dash, html, Input, Output, dcc, ctx, State
import dash_bootstrap_components as dbc
import module_data 

### Whatever the ACTIVE NODE is, it will be visually displayed and its information will be shown in the panel
def active_node_updates(app):
    @app.callback(Output('debugger', 'children'),
                 Input('hidden_active_module', 'children')
                )
    def filtering(data):
        if data:
            return data
