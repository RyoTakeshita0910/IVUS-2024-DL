import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2
import openpyxl

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, required=True, help='path to the folder containing origial pngs')
parser.add_argument("--out_xlsx", type=str, required=True, help='path to the output dataset folder')
parser.add_argument("--fold", type=int, default=5, help='k-fold cross validation')
args = parser.parse_args()

def make_id(name):
    name_split = name.split('_')
    name_id = ''
    for i in range(len(name_split)-1):
        name_id += name_split[i]
        if i < len(name_split)-2:
            name_id += '_'
    
    return name_id

def main():
    img_list = sorted(os.listdir(args.img_dir))
    out_list = []
    data_list = []
    data_cnt = 0
    for img in img_list:
        # img_id = img.split('_')[0]
        img_id = make_id(img)
        print(img_id)
        if not img_id in data_list:
            data_list.append(img_id)
            out_list.append([img_id])
            data_cnt += 1
    
    vcp = int(data_cnt/args.fold)
    vcp_ex = data_cnt % args.fold
    val_data = []
    for i in range(args.fold):
        val_cnt = 0
        for j,out in enumerate(out_list):
            dataname = out[0]
            if not dataname in val_data and val_cnt < vcp:
                val_data.append(dataname)
                out_list[j].append('val')
                val_cnt += 1

                if val_cnt == vcp:
                    if args.fold-i <= vcp_ex:
                        val_cnt -= 1
                        vcp_ex -= 1
            else:
                out_list[j].append('train')

    wb = openpyxl.Workbook()
    ws = wb.worksheets[0]
    for i in range(len(out_list)):
        for j in range(len(out_list[i])):
            ws.cell(row=i+1, column=j+1, value=out_list[i][j])
    wb.save(args.out_xlsx)

main()


