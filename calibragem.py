import json
import time
import tkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from gaze_tracking import GazeTracking
from datetime import datetime, timedelta
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
import random
import os
import copy

gaze = GazeTracking()

def obter_resolucao_tela():
    root = tkinter.Tk()
    root.withdraw()
    # print('w=',root.winfo_screenwidth())
    # print('h=',root.winfo_screenheight())
    return root.winfo_screenwidth(), root.winfo_screenheight()

def escalonar(pontos_x, pontos_y):
    res_x, res_y = obter_resolucao_tela()
    pontos_x = MinMaxScaler(feature_range=(0, res_x)).fit_transform([[x] for x in pontos_x])
    pontos_y = MinMaxScaler(feature_range=(0, res_y)).fit_transform([[y] for y in pontos_y])
    return pontos_x, pontos_y

def escalona_ponto(ponto_x, ponto_y):
    res_x, res_y = obter_resolucao_tela()
    return ponto_x*res_x, ponto_y*res_y

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def remover_outliers(pontos_x, pontos_y, metodo):

    pontos = []
    for i in range(0, len(pontos_x)):
        pontos.append(np.asarray([pontos_x[i], pontos_y[i]]))
    pontos = np.asarray(pontos)

    colunas = ['x', 'y']
    df = pd.DataFrame(data=pontos, columns=colunas)
    print('Shape antes da remocao:', df.shape)

    if metodo == "sem":
        return df['x'].to_numpy(), df['y'].to_numpy()

    elif metodo == "iqr":
        for col in colunas:
            q75,q25 = np.percentile(df.loc[:,col],[75,25])
            intr_qr = q75-q25
            max = q75+(1.5*intr_qr)
            min = q25-(1.5*intr_qr)
            df.loc[df[col] < min,col] = np.nan
            df.loc[df[col] > max,col] = np.nan

    elif metodo == "ecod":
        for col in colunas:
            data = df.loc[:,col].to_numpy().reshape(-1, 1)
            outliers = ECOD().fit_predict(data)
            outlier_indices = np.where(outliers == 1)[0]
            df.loc[outlier_indices,col] = np.nan

    elif metodo == "lof":
        for col in colunas:
            data = df.loc[:,col].to_numpy().reshape(-1, 1)
            outliers = LOF().fit_predict(data)
            outlier_indices = np.where(outliers == 1)[0]
            df.loc[outlier_indices,col] = np.nan

    print('Quantidade de outliers:', df.isna().sum())

    df = df.dropna(axis='index')

    print('Shape depois da remocao:', df.shape)

    return df['x'].to_numpy(), df['y'].to_numpy()

def draw_marker(x, y, radius = 20, color = (255, 0, 0)):
    # Criar uma janela vazia
    window_name = 'Marker'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Desenhar um círculo vermelho no ponto (x, y)
    thickness = -1  # Preenchido
    width, height = obter_resolucao_tela()
    bg_frame = np.zeros(shape=(height,width,3))
    cv2.circle(bg_frame, (int(x), int(y)), radius, color, thickness)

    # Exibir o marcador na tela
    cv2.imshow(window_name, bg_frame)
    cv2.waitKey(1)

def quad_center(i, j, n_row, n_col):
    width, height = obter_resolucao_tela()
    step_w = width / (n_col - 1)
    step_h = height / (n_row - 1)
    q_x = j * step_w
    q_y = i * step_h
    print(f'[quad_center] width={width} height={height} i={i} j={j} n_row={n_row} n_col={n_col} q_x={q_x} q_y={q_x}')
    return (q_x, q_y)



def plot_k_means(pontos_x, pontos_y, n_clusters, titulo="", escalonar=False):

    pontos = []
    for i in range(0, len(pontos_x)):
        pontos.append(np.asarray([pontos_x[i], pontos_y[i]]))
    pontos = np.asarray(pontos)

    km = KMeans(n_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=42)
    y_km = km.fit_predict(pontos)

    silhouette_avg = silhouette_score(pontos, y_km)
    calinski_harabasz_score_value = calinski_harabasz_score(pontos, y_km)
    print ("--------------------------------")
    print (titulo)
    print(f'Silhouette Score: {silhouette_avg:.6f}')
    print(f'Calinski-Harabasz Index: {calinski_harabasz_score_value:.6f}')
    # plot the n_clusters
    for i in range(0,n_clusters):
        plt.scatter(
            pontos[y_km == i, 0], pontos[y_km == i, 1],
            s=50, c=np.random.random(3),
            marker='s', edgecolor='black',
            label=f"cluster {i}"
        )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='black', edgecolor='black',
        label='centroids'
    )

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.title(titulo)
    
    image_path = 'resultados/'
    titulo_lower = titulo.lower()
    titulo_res = "-".join(titulo_lower.split())
    plt.savefig(image_path + titulo_res + ".png")

    try: 
        with open(f'{image_path}{titulo_res}.txt', 'x') as f:
            f.write(f'Silhouette Score: {silhouette_avg:.6f}\n')
            f.write(f'Calinski-Harabasz Index: {calinski_harabasz_score_value:.6f}')
    except FileExistsError:
        with open(f'{image_path}{titulo_res}.txt', 'w') as f:
            f.write(f'Silhouette Score: {silhouette_avg:.6f}\n')
            f.write(f'Calinski-Harabasz Index: {calinski_harabasz_score_value:.6f}')
            
    plt.show(block=True)

    


    #return (silhouette_avg, calinski_harabasz_score_value)


def generate_random_positions(n_row, n_col):
    all_positions = [(i,j) for i in range(n_row) for j in range(n_col)]
    random.shuffle(all_positions)
    return all_positions

def calibrate(n_row, n_col, delay, duration, type = 's',outlier_method="iqr"):
    radius = 30
    all_samples = [[0] * n_col] * n_row
    
    draw_marker(0,0,radius,(0,0,0))
    time.sleep(5)

    if (type == 's'):
        for i in range(n_row):
            for j in range(n_col):
                #print(f'row={i} col={j}')
                q_x, q_y = quad_center(i, j, n_row, n_col)

                if (j == 0):
                    q_x += radius
                elif (j == n_col-1):
                    q_x -= radius
                if (i == 0):
                    q_y += radius
                elif (i == n_row-1):
                    q_y -= radius

                draw_marker(q_x, q_y, radius, (255,255,255))
                time.sleep(delay)
                q_samples = capture(duration)
                all_samples[i][j] = q_samples
                #draw_marker(q_x, q_y, radius, (0,0,0))
               
                

    elif (type == 'r'):
        positions = generate_random_positions(n_row,n_col)
        for pos in positions:
            (i,j) = pos
            #print(f'row={i} col={j}')
            q_x, q_y = quad_center(i, j, n_row, n_col)

            if (j == 0):
                q_x += radius
            elif (j == n_col-1):
                q_x -= radius
            if (i == 0):
                q_y += radius
            elif (i == n_row-1):
                q_y -= radius

            draw_marker(q_x, q_y, radius, (255,255,255))
            time.sleep(delay)
            q_samples = capture(duration)
            all_samples[i][j] = q_samples
            #draw_marker(q_x, q_y, radius, (0,0,0))

            
    all_x, all_y = [], []
    for i in range(n_row):
        for j in range(n_col):
            for k in range(len(all_samples[i][j])):
                p_x, p_y = all_samples[i][j][k]
                all_x.append(p_x)
                all_y.append(p_y)

    n_clusters = n_col*n_row
    plot_k_means(pontos_x=all_x, pontos_y=all_y, n_clusters=n_clusters,titulo="Com outliers " + outlier_method + " " + type)

    """
    methods = {
        'iqr': {
            'all_x': [],
            'all_y': [],
            'silhouette': 0,
            'calisnky': 0,
        },
        'lof': {
            'all_x': [],
            'all_y': [],
            'silhouette': 0,
            'calisnky': 0,
        },
        'ecod': {
            'all_x': [],
            'all_y': [],
            'silhouette': 0,
            'calisnky': 0,
        },
        'sem': {
            'all_x': [],
            'all_y': [],
            'silhouette': 0,
            'calisnky': 0,
        },
    }

    for method, values in methods.items():
        values['all_x'] = copy.deepcopy(all_x)
        values['all_y'] = copy.deepcopy(all_y)
        
        all_x_copy, all_y_copy = remover_outliers(pontos_x=values['all_x'], pontos_y=values['all_y'], metodo = method)
        (silhouette, calisnky) = plot_k_means(pontos_x=all_x_copy, pontos_y=all_y_copy, n_clusters=n_clusters, titulo="Sem outliers " + method)
        values['silhouette'] = silhouette
        values['calisnky'] = calisnky
        values['all_x'] = all_x_copy
        values['all_y'] = all_y_copy

    with open(f'cache/methods.json', 'w') as f:
        json.dump(methods, f, indent=2, default=default)
    """
    all_x_copy, all_y_copy = copy.deepcopy(all_x), copy.deepcopy(all_y)
    all_x_copy, all_y_copy = remover_outliers(pontos_x=all_x_copy, pontos_y=all_y_copy, metodo = outlier_method)
    plot_k_means(pontos_x=all_x_copy, pontos_y=all_y_copy, n_clusters=n_clusters, titulo="Sem outliers " + outlier_method + " " + type)
    #for metodo in ['sem','ecod','lof','iqr']:
     #   all_x_copy, all_y_copy = copy.deepcopy(all_x), copy.deepcopy(all_y)
     #   all_x_copy, all_y_copy = remover_outliers(pontos_x=all_x_copy, pontos_y=all_y_copy, metodo = metodo)
     #   plot_k_means(pontos_x=all_x_copy, pontos_y=all_y_copy, n_clusters=n_clusters, titulo="Sem outliers " + metodo)

    return all_x_copy, all_y_copy

def capture(duration):
    webcam = cv2.VideoCapture(0)
    samples = []
    start_time = time.time()
    elapsed_time = 0
    while elapsed_time < duration:
        _, frame = webcam.read()
        gaze.refresh(frame)
        current_time = time.time()
        elapsed_time = current_time - start_time
        gaze_x = gaze.horizontal_ratio()
        gaze_y = gaze.vertical_ratio()
        if gaze_x != None and gaze_y != None:
            samples.append((1.0-gaze_x, gaze_y))
            print(f'x={(1.0-gaze_x):.3f}, y={(gaze_y):.3f}')
        time.sleep(0.01)
        # frame = gaze.annotated_frame()
    print(f'{len(samples)} amostras capturadas')
    return samples


def armazenar_lista_de_coordenadas(vet,nome):
    with open(f"cache/{nome}.json", 'w') as f:
        json.dump(vet, f, indent=2)

def obter_limites_webcam(valores, qtd=30):
    #print(f'entrou no metodo obter_limites()')
    # ordenar o vetor de forma crescente
    #vet = np.concatenate([v for v in valores.values()]).tolist()
    vet = valores
    vet.sort()
    # filtra o vetor removendo valores invalidos
    vet_filtered = [i for i in vet if i != None]
    # min = media dos 'qtd' primeiros valores
    v_min = np.mean(vet_filtered[:qtd])
    #print(f'v_minimo = {v_min}')
    # max = media dos 'qtd' ultimos valores
    v_max = np.mean(vet_filtered[-qtd:])
    #print(f'v_maximo = {v_max}')
    return v_min, v_max


def calibrar(type = 's', outlier_method = 'sem'):
    print(f'entrou no metodo calibragem()')
    lista_gaze_x, lista_gaze_y = calibrate(n_row=2, n_col=2, delay=1, duration=4, type=type,outlier_method=outlier_method)

    print(f'lista_gaze_x = {lista_gaze_x}')
    print(f'lista_gaze_y = {lista_gaze_y}')
    x_min,x_max = obter_limites_webcam(lista_gaze_x)
    y_min,y_max = obter_limites_webcam(lista_gaze_y)
    width,height = obter_resolucao_tela()
    ans = x_min,x_max,y_min,y_max,0,width,0,height
    with open(f'cache/calibragem.json', 'w') as f:
        json.dump(ans, f, indent=2)
    """
    for method, values in calibration_results.items():
        print(f'lista_gaze_x = {values["all_x"]}')
        print(f'lista_gaze_y = {values["all_y"]}')
        x_min,x_max = obter_limites_webcam(values['all_x'])
        y_min,y_max = obter_limites_webcam(values['all_y'])
        width,height = obter_resolucao_tela()
        ans = x_min,x_max,y_min,y_max,0,width,0,height
        with open(f'cache/calibragem_{method}.json', 'w') as f:
            json.dump(ans, f, indent=2)
    """
    return ans

