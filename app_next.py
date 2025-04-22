import numpy as np
import plotly.graph_objects as go # or plotly.express as px
from dash import Dash, dcc, Input, Output, State, callback, Patch, ctx, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify

import libmatrix as lm

csv_arr = lm.uploadFileByName('4_20.txt')
fig_matrix, deep_levels, max_matrix_size, max_order = lm.calcAllDatas(csv_arr)

#================================LAYOUT====================================
app = Dash(external_stylesheets=dmc.styles.ALL, title="Prognosis Ai.Next")


#===============HEADER COMPONENTS=============:
orderSelect = dmc.NumberInput(
    id="id-order",
    min=1,
    max=max_order,
    value=1,
    prefix='Order: ',
    leftSection=DashIconify(icon="healthicons:ui-menu-grid"),
    w=100,
    size='xs',
)

def setMatrixSizeComp(arr_size, hBack, new_size=256):
    arr_size -= hBack
    if new_size >= arr_size:
        new_size = arr_size
    return dmc.NumberInput(
        id="id-matrix-size",
        value=new_size,
        min=10,
        max=arr_size,
        prefix='Matrix size: ',
        leftSection=DashIconify(icon="healthicons:ui-menu-grid"),
        w=150,
        size='xs',
)
matrixSize=setMatrixSizeComp(arr_size=max_matrix_size, hBack=1)

historyBack = dmc.NumberInput(
    id="id-hist-back",
    min=1,
    value=1,
    prefix='H-back: ',
    leftSection=DashIconify(icon="healthicons:ui-menu-grid"),
    w=120,
    size='xs',
)

switchCalkMatrixF = dmc.Switch(
    size="xs",
    radius="md",
    label="Calc Matrix-F",
    labelPosition="left",
    checked=False,
    id="id-switch-calc-matrix-f"
)

fileUploadCard = dmc.HoverCard(
    width=240,
    withArrow=True,
    position='bottom-start',
    offset=10,
    children=[
        dmc.HoverCardTarget(dcc.Upload(
            dmc.Button('Upload File', size="xs", id="id-load-button"), id='id-upload')),
        dmc.HoverCardDropdown(
            [
                dmc.List(
                    [
                        dmc.ListItem(
                            [dmc.Text("File name: 4_20.txt")]),
                        dmc.ListItem(
                            [dmc.Text("Time calc datas: 0.16 sec")]),
                        dmc.ListItem(
                            [dmc.Text("Matrix max size: 800")]),
                        dmc.ListItem(
                            [dmc.Text("Orders: 4")]),
                        dmc.ListItem(
                            [dmc.Text("Max number: 20")]),
                    ],
                ),
            ],
        ),
    ],
    id="id-upload-card",
)

app_header = dmc.AppShellHeader(
    dmc.Group(
        [
            dmc.Title("Next.Ai", size="xs", c="blue"),
            fileUploadCard,
            switchCalkMatrixF,
            orderSelect,
            historyBack,
            matrixSize,
        ],
        h="ml",
        m=5,
    )
)

fig_chances = lm.drawChancesBar(20)

excluderMenu = dmc.Menu(
    [
        dmc.MenuTarget(dmc.Button("Excluder")),
        dmc.MenuDropdown(
            [
                dmc.MenuLabel("Exclude:"),
                dmc.MenuItem(
                    "Selected", leftSection=DashIconify(icon="tabler:settings")
                ),
                dmc.MenuItem(
                    "Unselected", leftSection=DashIconify(icon="tabler:message")
                ),
                dmc.MenuItem("Save", leftSection=DashIconify(icon="tabler:photo")),
                dmc.MenuDivider(),
                dmc.MenuLabel("Danger Zone"),
                dmc.MenuItem(
                    "Restore last",
                    leftSection=DashIconify(icon="tabler:arrows-left-right"),
                ),
                dmc.MenuItem(
                    "Restore all",
                    leftSection=DashIconify(icon="tabler:trash"),
                    color="red",
                ),
            ]
        ),
    ],
    trigger="hover",
)

app_main=dmc.AppShellMain(
    [
        dmc.Grid(
            [
                dmc.GridCol(
                    [
                        dmc.Stack(
                            [
                                dcc.Graph(figure=fig_chances, id="id-chances"),
                                excluderMenu,
                            ],
                            align="left",
                            gap="5",
                    )
                    ],
                    span=1
                ),
                dmc.GridCol(
                    [
                        dcc.Graph(figure=fig_matrix, id="id-matrix"),
                    ],
                    span=11
                )
            ],
            gutter=5,
            ml=5
        )     
    ],
)

layout = dmc.AppShell(
    [
        app_header,
        app_main,
    ],
    header={"height": "lg"},
)

app.layout = dmc.MantineProvider(
    [
        layout,
    ],
    forceColorScheme="dark",
)

#====================================CALLBACKS=============================
dcc.Upload()

@callback(
    # Установка компонентов по умолчанию для загруженного файла архива:
    Output('id-matrix', 'figure'),
    Output('id-matrix-size', 'max'),
    Output('id-matrix-size', 'value'),
    Output('id-order', 'max', 'value'),
    Output('id-order', 'value'),
    Output('id-hist-back', 'value'),
    #Input('id-load-button', 'n_clicks'),
    Input('id-upload', 'contents'),
    State('id-upload', 'filename'),
    prevent_initial_call=True,
)
def uploadFile(content, file_name): #TODO: update deepLevels???
    csv_arr = lm.read_csv(content)
    matrix_graf, _, max_matrix_size, max_order = lm.calcAllDatas(csv_arr)
    #print(csv_arr)
    print(file_name)
    h_back = 1
    return matrix_graf, max_matrix_size, 150, max_order, 1, h_back
#---------------------------------------------

@callback(
    Input('id-switch-calc-matrix-f', 'checked')
)
def recalcAllDatas(on_off):
    print(on_off)

#---------------------------------------------

@callback(
       Input('id-order', 'value'),
)
def selectOrder(order):
    print(order)

#---------------------------------------------

@callback(
    Output('id-matrix', 'figure', allow_duplicate=True),
    Input('id-matrix', 'clickData'),
    State('id-matrix', 'figure'),
    prevent_initial_call=True,
)
def onClickMatrix(xyz_coord, matrix_fig):
    #print(ctx.inputs)
    # Загружаем JSON-данные & извлекаем координаты матрицы x, y, z
    points = xyz_coord['points']
    for point in points:
        x = point['x']
        y = point['y'] #for select deep level's plot
        index_click = point['curveNumber']
    print(f"x: {x}, y: {y}) #, subfig: {index_click}")  #, z: {z}

    index_deep = 3
    """
    for i in range(len(matrix_fig['data'])):
        name = matrix_fig['data'][i]['name']
        if  name == "Event deep":
            index_deep = i
        elif name == "Click_Layer":
            index_click = i

    print("i_click: "+str(index_click))
    print("i_deep"+str(index_deep))
    """
    #patched_layer = Patch()
    #patched_layer = matrix_fig['data'][index_click]
    #print(patched_layer)

    #dots = matrix_fig['data'][index_deep]['selectedpoints']
    #print(dots)
    if index_click < 3:
        patched_level = Patch()
        patched_level["data"][index_deep]["y"] = deep_levels[:,y]
        return patched_level
    return no_update

    #print(patched_level)
    #print(patched_level['selectedpoints'])
#---------------------------------------------------------------

@callback(
    Output('id-matrix', 'figure', allow_duplicate=True),
    Input('id-matrix', 'selectedData'),
    State('id-matrix', 'figure'),
    prevent_initial_call=True,
)
def onSelectPoints(selected, matrix):
    print(selected)
    return no_update
    
if __name__ == "__main__":
    app.run(debug=True)