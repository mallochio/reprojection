from cgitb import text
import PySimpleGUI as sg
import sys
import os.path
from glob import glob
import cv2
import numpy as np

from utils.framekeeper import FrameKeeper

def check_selected_folder(route):
    err = []
    captures = glob(f'{route}/capture*')
    omni = glob(f'{route}/omni')
    if len(omni) != 1:
        err.append('Omni folder missing')
    
    synced = False
    if os.path.exists(f'{route}/shots.txt') or os.path.exists(f'{route}/syncro_data.json'):
        synced = True
    
    calib = False
    calibfiles = glob(f'{route}/omni*Params.json')
    if len(calibfiles) == len(captures):
        calib = True

    dp_data = False
    dp_folders = glob(f'{route}/capture*/rgb_dp2_*')
    if len(dp_folders) == 2*len(captures):
        dp_data = True
    elif len(dp_folders) == 0:
        dp_data = False
    else:
        sg.popup_error('Some Densepose data folders are missing.')
        err.append('Some densepoe folders missing.')

    return {
        'errors': err, 'omni': len(omni),
        'captures': len(captures), 'synced': synced,
        'calib': calib, 'densepose': dp_data,
        'route': route}


def show_pick_folder_window():

    layout = [
        [sg.Text('Working folder:'), sg.Input(key='txtFolder', disabled=True, enable_events=True), sg.FolderBrowse('Browse', key='btBrowse')],
        [sg.Text('Summary:', font='Arial 16')],
        [sg.Text('Kinect captures:'), sg.Text('', key='txtKinects')],
        [sg.Checkbox(" Sync'ed",key='chkSync', disabled=True)],
        [sg.Checkbox(" Calibrated kinects wrt omni",key='chkCalib', disabled=True)],
        [sg.Checkbox(" Densepose masks",key='chkDP', disabled=True)],
        [sg.Text('Options:', font='Arial 16')],
        [sg.Radio('Colorize per kinect id', 'kc', key='kcId', default=True)],
        [sg.Radio('Colorize using kinect RGB pixel values', 'kc', key='kcRGB', disabled=True)],
        [sg.Radio('Colorize using Densepose body part indices', 'kc', key='kcDP', disabled=True)],
        [sg.Button('Continue', key='btCont'), sg.Button('Exit', key='btExit')]
        ]

    pick_folder_window = sg.Window('Reprojection GUI: Select working folder', layout=layout, modal=True)

    while True:
        event, values = pick_folder_window.read()
        if event == 'txtFolder':
            route = values['txtFolder']  # sg.popup_get_folder('Pick a working folder', 'Select folder')
            results = check_selected_folder(route)
            if len(results['errors']) != 0:
                sg.popup_error(f"Errors were found in folder check: {results['errors']}")
                continue

            pick_folder_window['txtFolder'].update(route)
            pick_folder_window['txtKinects'].update(results['captures'])
            pick_folder_window['chkSync'].update(results['synced'])
            pick_folder_window['chkCalib'].update(results['calib'])
            pick_folder_window['chkDP'].update(results['densepose'])

            pick_folder_window['kcDP'].update(disabled = not results['densepose'])

        if event == sg.WIN_CLOSED or event == 'btExit':
            pick_folder_window.close()
            sys.exit(0)

        if event == 'btCont':
            if values['txtFolder'] == '':
                sg.popup_error('Please select folder first.')
            elif values['chkCalib'] == False:
                sg.popup_error('Please calibrate set first.')
            
            pick_folder_window.close()
            break
    
    return results


def show_main_window(results = {}):
    fk = None
    span = []
    ts = 0
    menu_def = [
        ['&File', ['&Open folder', '---', '&Quit']],
        ['Options', ['&Synchronize ...', '&Reproject ...']]
        ]

    main_layout = [
        [sg.MenuBar(menu_def)],
        [sg.Text('Working path:'), sg.Input(default_text='', key='txtRoute', disabled=True),
         sg.Text('Frame:'), sg.Input(size=(6,), key='txtFrame'), sg.Button('Go'), sg.Text('N. frames:'), sg.Input(size=(6,), key='txtNframes', disabled=True)],
        [sg.Image(size=(600,400), key='k0', background_color='darkblue'), sg.Image(size=(600,400), key='k1', background_color='darkblue'), sg.Image(size=(600,400), key='k2', background_color='darkblue')],
        [sg.Image(size=(600,400), key='omni', background_color='darkblue'), sg.Image(size=(600,400), key='repro', background_color='darkblue'), sg.Image(size=(600,400), key='opt', background_color='darkblue')],
        [sg.Button('Play'), sg.Button('Pause'), sg.Button('Prev. frame'), sg.Button('Next frame'), sg.Button('< 100ms'), sg.Button('100ms >'), sg.Button('< 1s'), sg.Button('1s >')]
        ]

    main_window = sg.Window('Reprojection Suite GUI', main_layout)

    while True:
        event, values = main_window.read()

        if event == 'Quit' or event == sg.WIN_CLOSED:
            main_window.close()
            break
        
        if event == 'Open folder':
            results = show_pick_folder_window()
            
            main_window['txtRoute'].update(results['route'])
            fk = FrameKeeper(results['route'], capture_Hz=15)
            span = fk.get_lead_span()
            main_window['txtNframes'].update(f'{len(span)}')
            main_window['txtFrame'].update('0') 
            ts = span[0]
            frames, _ = fk.get_syncro_frames(ts)
            for k in range(fk.num_kinects):
                depth = cv2.flip(frames[f'capture{k}'], 1)
                depth = cv2.undistort(depth, fk.kinect_params[k]['K'], fk.kinect_params[k]['D'])
                depth = np.uint8((depth / 4500.) * 255.)
                depth = cv2.resize(depth, dsize=(482,400))
                h, w = depth.shape
                shown = np.zeros((400,600), np.uint8)
                H, W = shown.shape
                shown[H//2-h//2:H//2+h//2,W//2-w//2:W//2+w//2] = depth
                png_bytes = cv2.imencode('.png', shown)[1].tobytes()
                main_window[f'k{k}'](data=png_bytes)
            
            omni = frames['omni']
            omni = cv2.resize(omni, dsize=(600,400))
            png_bytes = cv2.imencode('.png', omni)[1].tobytes()
            main_window['omni'](data=png_bytes)



def main():
    #results = show_pick_folder_window()
    show_main_window()

    
if __name__ == '__main__':
    main()