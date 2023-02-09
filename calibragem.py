import json
import time
import tkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from gaze_tracking import GazeTracking


def obter_resolucao_tela():
    root = tkinter.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

def escalonar(pontos_x, pontos_y):
    res_x, res_y = obter_resolucao_tela()
    pontos_x = MinMaxScaler(feature_range=(0, res_x)).fit_transform([[x] for x in pontos_x])
    pontos_y = MinMaxScaler(feature_range=(0, res_y)).fit_transform([[y] for y in pontos_y])
    return pontos_x, pontos_y

def remover_outliers(pontos_x, pontos_y):

    pontos = []
    for i in range(0, len(pontos_x)):
        pontos.append(np.asarray([pontos_x[i], pontos_y[i]]))
    pontos = np.asarray(pontos)

    colunas = ['x', 'y']
    df = pd.DataFrame(data=pontos, columns=colunas)

    for col in colunas:
        q75,q25 = np.percentile(df.loc[:,col],[75,25])
        intr_qr = q75-q25

        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)

        df.loc[df[col] < min,col] = np.nan
        df.loc[df[col] > max,col] = np.nan

    df = df.dropna(axis=0)

    return df['x'].to_numpy(), df['y'].to_numpy()

def plotar_k_means(pontos_x, pontos_y, escalonar=False):

    pontos = []
    for i in range(0, len(pontos_x)):
        pontos.append(np.asarray([pontos_x[i], pontos_y[i]]))
    pontos = np.asarray(pontos)

    km = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=42)
    y_km = km.fit_predict(pontos)

    # plot the 3 clusters
    plt.scatter(
        pontos[y_km == 0, 0], pontos[y_km == 0, 1],
        s=50, c='#d7191c',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        pontos[y_km == 1, 0], pontos[y_km == 1, 1],
        s=50, c='#fdae61',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        pontos[y_km == 2, 0], pontos[y_km == 2, 1],
        s=50, c='#abdda4',
        marker='v', edgecolor='black',
        label='cluster 3'
    )

    plt.scatter(
        pontos[y_km == 3, 0], pontos[y_km == 3, 1],
        s=50, c='#2b83ba',
        marker='v', edgecolor='black',
        label='cluster 3'
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
    plt.show(block=True)

def capturar_cantos_webcam():
    print(f'entrou no metodo capturar_cantos()')
	
    print(f'Initializing calibration.')
    start_delay = 3  # seconds
    print(f'Calibration will start in {start_delay} seconds.')
    time.sleep(start_delay)

    time_delta = 0
    elapsed_time = 0
    temp_ini = time.time()  # tempo que começa o programa
    start_time = time.time()
    print('Main loop started.')

    window_name = "Gaze Tracking"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    lista_gaze_x = []
    lista_gaze_y = []
    elapsed_time = 0
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    WEBCAM_WIDTH = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    WEBCAM_HEIGHT = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'WEBCAM_WIDTH = {WEBCAM_WIDTH}, WEBCAM_HEIGHT = {WEBCAM_HEIGHT}')
    blank_image = np.zeros((WEBCAM_HEIGHT,WEBCAM_WIDTH,3), dtype=np.uint8) + 255

    curr_pos = 'nenhuma'
    start_time = time.time()  # inicia tempo dentro do while

    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        # get the x,y coordinate for the right pupil
        right_pupils_coord = gaze.pupil_right_coords()

        current_time = time.time()
        elapsed_time = current_time - start_time

        gaze_x = gaze.horizontal_ratio()
        gaze_y = gaze.vertical_ratio()

        if gaze_x != None and gaze_y != None:

            # +------C------+
            # |             |
            # E             D
            # |             |
            # +------B------+

            last_pos = curr_pos
            if elapsed_time < 5.0:
                curr_pos = 'esquerda'
            elif 5.0 <= elapsed_time < 10.0:
                curr_pos = 'cima'
            elif 10.0 <= elapsed_time < 15.0:
                curr_pos = 'direita'
            elif 15.0 <= elapsed_time < 20.0:
                curr_pos = 'baixo'
            else:
                print(f'Main loop finished after {str(int(elapsed_time))} seconds.')
                break
            current_img = cv2.imread(f'./assets/img_{curr_pos}.png')
            print(right_pupils_coord)
            cv2.circle(current_img, (right_pupils_coord if right_pupils_coord else (0,0)), radius=10, color=(255, 0, 0), thickness=-1)
            cv2.imshow(window_name, current_img)

          #  if curr_pos != last_pos:
            #    print(f'desenhando')
             #   cv2.imshow(window_name, current_img)

            print(f'elapsed_time = {elapsed_time} last_pos = {last_pos} curr_pos = {curr_pos}')

            lista_gaze_x.append(gaze_x)
            lista_gaze_y.append(gaze_y)

            time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f'Main loop finished after {str(int(elapsed_time))} seconds.')
            break

    webcam.release()
    cv2.destroyAllWindows()

    # lista_gaze_x, lista_gaze_y = escalonar(lista_gaze_x, lista_gaze_y)

    lista_gaze_x, lista_gaze_y = remover_outliers(pontos_x=lista_gaze_x, pontos_y=lista_gaze_y)

    plotar_k_means(pontos_x=lista_gaze_x, pontos_y=lista_gaze_y)

    return lista_gaze_x, lista_gaze_y


def obter_limites_webcam(vet, qtd=3):
    print(f'entrou no metodo obter_limites()')
    # ordenar o vetor de forma crescente
    vet.sort()
    # filtra o vetor removendo valores invalidos
    vet_filtered = [i for i in vet if i != None]
    # min = media dos 'qtd' primeiros valores
    v_min = np.mean(vet_filtered[:qtd])
    print(f'v_minimo = {v_min}')
    # max = media dos 'qtd' ultimos valores
    v_max = np.mean(vet_filtered[-qtd:])
    print(f'v_maximo = {v_max}')
    return v_min, v_max


def calibrar():
    print(f'entrou no metodo calibragem()')
    lista_gaze_x,lista_gaze_y = capturar_cantos_webcam()
    x_min,x_max = obter_limites_webcam(lista_gaze_x)
    y_min,y_max = obter_limites_webcam(lista_gaze_y)
    width,height = obter_resolucao_tela()
    ans = x_min,x_max,y_min,y_max,0,width,0,height
    with open("cache/calibragem.json", 'w') as f:
        json.dump(ans, f, indent=2)
    return ans