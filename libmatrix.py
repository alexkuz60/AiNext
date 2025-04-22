import time
import base64
import numpy as np
import plotly.graph_objects as go # or plotly.express as px
from plotly.subplots import make_subplots

#---------------------------------------------------------
def read_csv(content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    csv_data = decoded.decode('utf-8-sig').splitlines()
    arr_data = np.loadtxt(csv_data, delimiter='\t').astype(int)

    return arr_data

def calcAllTracks(input_arr, matrix_size, cols):
    #st=time.time()
    # Создаем one-hot матрицу за одну операцию
    tracks_events = np.zeros((matrix_size, cols), dtype=np.int32)  # Используем меньший тип данных
    np.put_along_axis(tracks_events, (input_arr - 1)[:, None], 1, axis=1)
    
    # Накопительная сумма с явным указанием выходного типа
    cumulative_data = np.cumsum(tracks_events, axis=0, dtype=np.int32)
    #print(time.time()-st)
    
    return tracks_events, cumulative_data

def getWinTrack(tracks, order_data, h_back):
    #all_tracks = np.argwhere(tracks == 1)
    #print(all_tracks)
    if h_back > 0:
        n = order_data[0] - 1
        track = tracks[:, n]
    return track
#---------------------------------------------------------

def extract_track_xy(separate_data, cumulative_data, track_num):
    """
    Возвращает для одного трека (track_num, 1-based) массив shape=(2, n+2),
    где первая строка — X‑координаты, вторая — Y‑координаты.
    """
    tn = track_num - 1
    idx = np.flatnonzero(separate_data[:, tn])
    n = idx.size

    x = np.concatenate(([0], idx, [cumulative_data.shape[0] - 1]))
    if n > 0:
        y_vals = cumulative_data[idx, tn]
        y = np.concatenate(([0], y_vals, [y_vals[-1]]))
    else:
        y = np.zeros(2, dtype=int)  # только [0, 0], если нет точек
        x = np.array([0, cumulative_data.shape[0] - 1], dtype=int)
    return np.vstack((x, y))
#-----------------------------------------------------------

def extract_all_tracks(separate_data, cumulative_data):
    """
    Возвращает список и словарь всех треков.
    all_tracks[i] — результат для track_num=i+1
    all_tracks_dict[track_num] — результат для track_num
    """
    n_tracks = cumulative_data.shape[1]
    all_tracks = [extract_track_xy(separate_data, cumulative_data, i+1) for i in range(n_tracks)]
    #all_tracks_dict = {i+1: arr for i, arr in enumerate(all_tracks_list)}

    return all_tracks
#---------------------------------------------------------

#----------------------------------------------------------

def calcPassport(matrix_data, max_deep):
    #st=time.time()
    # Альтернативная реализация с flat-индексами
    rows = np.repeat(np.arange(matrix_data.shape[0]), matrix_data.shape[1])
    cols = matrix_data.ravel()
    passport = np.zeros((matrix_data.shape[0], max_deep), dtype=int)
    np.add.at(passport, (rows, cols), 1)
    #print(time.time()-st)

    return passport

#----------------------------------------------------------

def calcDeepLevels(passport, win_track, max_deep):
    #st=time.time()
    # next_val: для каждой строки, это passport, сдвинутый на 1 влево, а последний столбец — 0
    next_val = np.zeros_like(passport)
    next_val[:, :-1] = passport[:, 1:]  # для всех, кроме последнего столбца

    # event: 1 если win_track совпадает с глубиной, иначе 0
    depth_indices = np.arange(max_deep)
    event = (win_track[:, None] == depth_indices).astype(int)

    levels = passport - next_val - event
    #print(time.time()-st)
    return levels

#----------------------------------------------------------

def createMatrixGraf(data_h, calcF, data_f, click_layer, tracks, n_track, levels, deep):
    #max_range = len(tracks)-1
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.4],
        shared_xaxes=True,
        subplot_titles=["Tracks Matrix", "Events Deep Level"]
    )
    fig.add_heatmap(
        name="Matrix-H",
        z=data_h,
        xgap=1,
        ygap=1,
        hoverongaps=False,
        opacity=0.75,
        #colorscale='rainbow',
        showscale=False,
        showlegend=True,
        row=1, col=1
    )
    if calcF: #Draw Matxrix-F
        fig.add_heatmap(
            name="Matrix-F",
            z=data_f,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            opacity=0.75,
            showscale=False,
            showlegend=True,
            
            row=1, col=1
        )
    fig.add_heatmap( #Draw Click Layer
        name="Click_Layer",
        z=click_layer,
        xgap=1,
        ygap=1,
        hoverongaps=True,
        showscale=False,
        row=1, col=1
    )
    fig.add_scatter( #Draw win Track
        name="Win track",
        #y = tracks[:, n_track],
        x=tracks[0],
        y=tracks[1],
        showlegend=True,
        line_shape='hv',
        row=1, col=1
    )
    fig.add_scatter(
        name="Event deep",
        y=levels[:,deep],
        marker_size=10,
        mode='lines+markers',
        fill='tozerox',
        fillcolor="rgba(127,0,255, 0.25)",
        line_shape='spline',
        showlegend=False,
        row=2, col=1
    )
    #fig.update_traces(overwrite=True, selector=)
    fig.update_layout(
        template="plotly_dark",
        hoversubplots = "overlaying",
        #hovermode="x",
        clickmode='event+select',
        height = 900,
        margin=dict(t=150, l=20, r=20),
        coloraxis_showscale=False,
        legend = dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            orientation="h"),
        xaxis=dict(
            #autorange=True,
            #range=[0, max_range],
            rangeslider=dict(visible=True, thickness=0.1),
            #rangeslider_thickness=0.1,
            showspikes=True,
            spikemode='across',
            showgrid=True,
            tickfont_size = 14,
        ),
        xaxis2=dict(
            #range=[0, max_range],
            showspikes=True,
            spikemode='across',
            tickfont_size = 14,
        ),
        yaxis2=dict(
            showspikes=True,
            spikemode='across',
        ),
        yaxis=dict(
            title=dict(
                text="Events deep",
                font=dict(
                    size=16
                )
            ),
            autorange="reversed",
            showspikes=True,
            spikemode='across',
            showgrid=True,
            tickfont_size = 14,
        ),
    )

    return fig

#----------------------------------------------
def uploadFileByName(input_file):
    # Статовая загрузка CSV файла по умолчанию в массив NumPy
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        csv_data = np.loadtxt(f, delimiter='\t').astype(int)
    return csv_data
#--------------------------------------------------------------

def calcAllDatas(csv_data, calc_f=False, order=1, hist_back=1):
    start = time.time()

    # Загрузка CSV файла в массив NumPy
    #with open(input_file, 'r', encoding='utf-8-sig') as f:
    #    csv_data = np.loadtxt(f, delimiter='\t').astype(int)    # skiprows=1, если есть заголовок

    max_order = csv_data.shape[1]
    max_num = csv_data.max()
    matrix_size = csv_data.shape[0] - hist_back - 1
    order_data_h = csv_data[hist_back : (matrix_size + hist_back), order]
    
    len_csv = len(order_data_h)

    tracks, arr_matrix_h = calcAllTracks(input_arr=order_data_h, matrix_size=len_csv, cols=max_num)
        #print('----Tracks cumulative:')
    #Selected track for view:
    #test:
    tracks=extract_all_tracks(tracks, arr_matrix_h)
    track=tracks[0]

    if calc_f:
        # Инверсная Матрица (из прошлого в будущее)
        order_data_f = order_data_h[::-1]
        #print(order_data_f)
        
        #print(arr_matrix)
        arr_matrix_f = calcAllTracks(input_arr=order_data_f, matrix_size=len_csv, cols=max_num)

    
    # WinTrack data:
    win_track_h = getWinTrack(tracks=arr_matrix_h, order_data=order_data_h, h_back=hist_back)

    # Определяем максимальное значение таблицы треков
    max_deep = np.max(arr_matrix_h) + 1
    #-------------Подготовка матрицы 
    passport_h = calcPassport(arr_matrix_h, max_deep)
    #print('-----Passport:')
    #print(passport_arr)
    

    # Таблица динамических графиков для всех уровней глубины:
    deep_levels = calcDeepLevels(passport_h, win_track_h, max_deep)
    #print(deep_levels)

    #-------------Подготовка матрицы для графика:
    #Замена нулей в матрице на None
    matrix_h = passport_h
    matrix_h = np.where(passport_h==0, np.nan, passport_h)
    
    #test:

    # Разворот матрицы для графика матрицы:
    matrix_h = np.rot90(matrix_h, k=1)
    matrix_h = matrix_h[::-1]
    #print(matrix_h)

    #Создание клик-матрицы:
    click_matrix = np.full_like(matrix_h, np.nan)
    #print(click_matrix)

    if calc_f:
        #-----------------Инверсная матрица:
        passport_f = calcPassport(arr_matrix_f, max_deep)
        #-----Подготовка матрицы для графика:
        #Замена нулей в матрице на None
        matrix_f = passport_f
        matrix_f = np.where(matrix_f==0, np.nan, matrix_f)
        # Разворот матрицы для графика матрицы:
        matrix_f = np.rot90(matrix_f, k=-1)
        #matrix_f = matrix_f[::-1]
    else:
        matrix_f=[]
    
    matrix_graf = createMatrixGraf(matrix_h, calc_f, matrix_f, click_matrix, track, n_track=0, levels=deep_levels, deep=2)

    print (time.time() - start)
    
    return matrix_graf, deep_levels, matrix_size, max_order
#---------------------------------------------------------------
def calcChances(cum_nums, passport, click_matrix, exeptedNums):
    clicksXY = np.argwhere(click_matrix > 0)
    #print(clicksXY)
        
#--------------------------------------------------------------
def drawChancesBar(max_num):
    max = max_num+1,
    nums = list(range(1, 51))   #np.arange(1, max, dtype=int)
    vals = np.random.randint(0, 100, size=max)
    fig = go.Figure(layout_template="plotly_dark")
    #print(vals)
    fig.add_bar(
        x=vals,
        y=nums,
        orientation='h',
    )
    fig.update_layout(
        #title_text="Events chances",
        clickmode='event+select',
        height = 800,
        margin=dict(t=150, l=10, r=10, b=10)
    )
    return fig


